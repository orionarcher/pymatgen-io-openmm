
PARAMETERIZER_TYPE_REQUIRED = "Explicit Parameterizer Assignment Requires Parameterizer Type to be set"

def UNEXPECTED_TOPOLOGY_TYPE(parametizer_type, actual, expected):
    return f"{parametizer_type.name} requires topology of type {expected} but got type {actual}"