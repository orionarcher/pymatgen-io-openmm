
PARAMETERIZER_TYPE_REQUIRED = "Explicit Parameterizer Assignment Requires Parameterizer Type to be set"
INVALID_PARAMETERIZER_TYPE = "Parameterizer Type Invalid"
INVALID_ASSIGNMENT_TYPE = "Assignment Type Invalid"
MULTIPLE_PARAMETERIZERS_NOT_SUPPORTED = "All force fields in configuration must have the same parameterizer type"
FORCE_FIELD_CUSTOMIZATION_NOT_SUPPORTED = "Force field customization only supported for OpenMM parameterizer types"
MISSING_CUSTOM_FORCE_FIELD_FILES = "Force Field customization requires > 1 customization filed path but was provided None"

def UNEXPECTED_TOPOLOGY_TYPE(parametizer_type, actual, expected):
    return f"{parametizer_type.name} requires topology of type {expected} but got type {actual}"

def UNSUPPORTED_FORCE_FIELD_TYPE(parameterizer_type):
    return f"{parameterizer_type.name} only supports the following force fields ${parameterizer_type.value.keys()}"
