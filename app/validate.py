"""Flag identifiers in generated Three.js code that the library does not have.

The model invented "renderer.setRenderMode()" and "WEBGL_RENDER_ALLOCATION".
That failure mode is checkable without a GPU, without a retrain and without a
teacher model: the training corpus already contains the entire three.js source,
so the real API surface can be extracted once and used as a reference.

What this catches
  * THREE.Something            - not a class, constant or global the library exports
  * new THREE.Something( )     - same, in constructor position
  * object.someMethod( )       - a method name that appears nowhere in the library
  * SOME_CONSTANT              - an ALL_CAPS identifier that is not a known constant

What this deliberately does NOT do
  * It does not claim a flagged call is wrong. A user-defined method on a
    user-defined class is legitimate. Findings are ADVISORY: they say "three.js
    has no such name", which is exactly the evidence a reader needs.
  * It does not validate semantics, argument counts, or ordering - only names.

A method name is accepted if it appears on ANY class in the library, because
per-class attribution is unreliable from regex extraction and would produce
false positives. The union is what makes "setRenderMode" detectable: it exists
on no class at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SURFACE = Path(__file__).resolve().parent.parent / "artifacts" / "api_surface.json"

_CACHE: dict = {}

# three.js members that are re-exported through the THREE namespace but are not
# classes in src/, plus JS/DOM members that legitimately appear on these objects.
EXTRA_ALLOWED = {
    "x", "y", "z", "w", "r", "g", "b", "a", "u", "v",
    "width", "height", "depth", "length", "count", "size", "value", "name",
    "type", "id", "uuid", "parent", "children", "position", "rotation", "scale",
    "quaternion", "matrix", "matrixWorld", "visible", "castShadow",
    "receiveShadow", "frustumCulled", "renderOrder", "userData", "layers",
    "geometry", "material", "materials", "texture", "map", "color", "opacity",
    "transparent", "side", "roughness", "metalness", "emissive", "normalMap",
    "roughnessMap", "metalnessMap", "aoMap", "envMap", "displacementMap",
    "alphaMap", "lightMap", "bumpMap", "clearcoat", "transmission", "thickness",
    "ior", "sheen", "iridescence", "anisotropy", "flatShading", "wireframe",
    "vertexColors", "fog", "background", "environment", "environmentIntensity",
    "intensity", "distance", "decay", "angle", "penumbra", "groundColor",
    "shadow", "camera", "target", "near", "far", "fov", "aspect", "zoom",
    "domElement", "shadowMap", "enabled", "autoClear", "info", "capabilities",
    "xr", "outputColorSpace", "toneMapping", "toneMappingExposure",
    "setAnimationLoop", "render", "dispose", "clone", "copy", "traverse",
    "add", "remove", "clear", "getObjectByName", "lookAt", "translateX",
    "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "updateMatrix",
    "updateMatrixWorld", "updateProjectionMatrix", "computeBoundingSphere",
    "toJSON", "onBeforeRender", "onAfterRender", "isMesh", "isObject3D",
    "isMaterial", "isTexture", "isBufferGeometry", "isLight", "isCamera",
    # DOM / JS
    "addEventListener", "removeEventListener", "appendChild", "getElementById",
    "querySelector", "querySelectorAll", "createElement", "setAttribute",
    "getContext", "requestAnimationFrame", "cancelAnimationFrame",
    "toFixed", "toString", "forEach", "map", "filter", "reduce", "push", "pop",
    "slice", "splice", "join", "split", "replace", "indexOf", "includes",
    "keys", "values", "entries", "assign", "freeze", "from", "isArray",
    "parse", "stringify", "min", "max", "abs", "sin", "cos", "tan", "sqrt",
    "pow", "floor", "ceil", "round", "random", "hypot", "atan2", "sign",
    "toLowerCase", "toUpperCase", "trim", "startsWith", "endsWith", "padStart",
    "now", "log", "warn", "error", "table", "assert", "group", "time", "timeEnd",
    "pow", "clamp", "lerp", "set", "setHSL", "setRGB", "getHex", "getHexString",
    "setScalar", "setFromMatrixPosition", "applyMatrix4", "normalize", "cross",
    "dot", "distanceTo", "addScaledVector", "multiplyScalar", "divideScalar",
    "lookAt", "getWorldPosition", "localToWorld", "worldToLocal", "rotateOnAxis",
}

METHOD_CALL_RE = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\.\s*"
                            r"([A-Za-z_$][\w$]*)\s*\(")
THREE_MEMBER_RE = re.compile(r"\bTHREE\s*\.\s*([A-Za-z_$][\w$]*)")
ALL_CAPS_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")

# Only a CLASS or FUNCTION declaration in the same snippet makes a target's
# methods off-limits. An earlier version also matched "const renderer = ...",
# which meant every ordinary object was exempt - and that silently swallowed the
# very hallucination this tool exists to catch (renderer.setRenderMode).
LOCAL_DECL_RE = re.compile(r"\b(?:class|function)\s+([A-Za-z_$][\w$]*)")

# Identifiers that are never three.js and must not be reported.
JS_GLOBALS = {
    "THREE", "Math", "Object", "Array", "JSON", "window", "document", "console",
    "performance", "Number", "String", "Boolean", "Date", "Promise", "Map",
    "Set", "WeakMap", "RegExp", "Error", "Symbol", "BigInt", "NaN", "Infinity",
    "URL", "Blob", "File", "FormData", "Image", "Audio", "Video", "Node",
    "Element", "HTMLElement", "Event", "HTMLElement", "Uint8Array",
    "Uint16Array", "Uint32Array", "Int8Array", "Int16Array", "Int32Array",
    "Float32Array", "Float64Array", "ArrayBuffer", "DataView", "TextEncoder",
    "TextDecoder", "Worker", "WebSocket", "XMLHttpRequest", "fetch", "WebGL2RenderingContext",
    "GPUDevice", "GPUAdapter", "navigator", "location", "history", "screen",
    "requestAnimationFrame", "cancelAnimationFrame", "setTimeout",
    "clearTimeout", "setInterval", "clearInterval", "parseInt", "parseFloat",
    "isNaN", "isFinite", "encodeURIComponent", "decodeURIComponent",
    "structuredClone", "queueMicrotask", "globalThis", "self", "top", "parent",
    "localStorage", "sessionStorage", "indexedDB", "crypto", "CSS", "GUI",
    "THREE",
}


def load_surface() -> dict:
    if "data" not in _CACHE:
        if not SURFACE.exists():
            raise FileNotFoundError(
                f"{SURFACE} missing - run train/12_api_surface.py first")
        data = json.loads(SURFACE.read_text(encoding="utf-8"))
        methods = set(EXTRA_ALLOWED)
        for entry in data["classes"].values():
            methods.update(entry["methods"])
        _CACHE["data"] = data
        _CACHE["classes"] = set(data["classes"])
        _CACHE["globals"] = set(data["globals"])
        _CACHE["methods"] = methods
        _CACHE["constants"] = set(data["globals"]) | {
            g for g in data["globals"] if g.isupper() or g[0].isupper()}
    return _CACHE


def validate_code(code: str) -> dict:
    """Return advisory findings about unknown three.js identifiers."""
    surface = load_surface()
    classes = surface["classes"]
    globals_ = surface["globals"]
    methods = surface["methods"]
    constants = surface["constants"]

    known_members = classes | globals_

    unknown_three = []
    seen = set()
    for match in THREE_MEMBER_RE.finditer(code):
        name = match.group(1)
        if name in known_members or name in methods or name in seen:
            continue
        seen.add(name)
        unknown_three.append(name)

    # Targets that never carry our API: JS built-ins and the THREE namespace.
    LOCAL_DECLS_SKIP = JS_GLOBALS

    local_decls = set(LOCAL_DECL_RE.findall(code))

    unknown_methods = []
    seen_methods = set()
    call_count = 0
    for match in METHOD_CALL_RE.finditer(code):
        target, method = match.group(1), match.group(2)
        call_count += 1
        if method in methods or method in seen_methods:
            continue
        # "THREE.Foo(" is a namespace member, already covered above.
        if target in LOCAL_DECLS_SKIP or target in local_decls:
            continue
        seen_methods.add(method)
        unknown_methods.append(f"{target}.{method}")

    unknown_constants = []
    seen_constants = set()
    for match in ALL_CAPS_RE.finditer(code):
        name = match.group(1)
        if (name in constants or name in seen_constants or name in methods
                or name in JS_GLOBALS):
            continue
        seen_constants.add(name)
        unknown_constants.append(name)

    findings = (len(unknown_three) + len(unknown_methods)
                + len(unknown_constants))
    return {
        "unknown_three_members": unknown_three,
        "unknown_methods": unknown_methods,
        "unknown_constants": unknown_constants,
        "method_calls_checked": call_count,
        "findings": findings,
        "verdict": ("clean" if findings == 0
                    else "suspicious" if findings <= 3 else "hallucinated"),
    }


if __name__ == "__main__":
    import sys

    sample = sys.stdin.read() if len(sys.argv) == 1 else Path(sys.argv[1]).read_text(
        encoding="utf-8")
    print(json.dumps(validate_code(sample), indent=2))
