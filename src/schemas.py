
REQUIRED_FIELDS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country"
]


def validate_input(data):

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object.")

    missing = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    extra = [
        field for field in data
        if field not in REQUIRED_FIELDS
    ]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    if extra:
        raise ValueError(f"Unexpected fields: {extra}")

    for field in REQUIRED_FIELDS:
        if data[field] is None:
            raise ValueError(f"Field cannot be null: {field}")

    numeric_fields = [
        "age",
        "fnlwgt",
        "education.num",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    for field in numeric_fields:
        try:
            data[field] = float(data[field])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value: {field}")

    return data
