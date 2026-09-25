
def parse_log_line(line):
    """
    Parse one synthetic QXDM-style modem log line.

    Returns:
        dict: Structured log record
        None: If the line is invalid
    """

    line = line.strip()

    if not line:
        return None

    parts = line.split("|")

    if len(parts) < 5:
        return None

    timestamp = parts[0].strip()
    rat = parts[1].strip()
    protocol = parts[2].strip()
    event = parts[3].strip()

    ue_field = parts[4].strip()

    if "=" not in ue_field:
        return None

    ue_id = ue_field.split("=", 1)[1]

    cause = None
    count = None

    for field in parts[5:]:

        field = field.strip()

        if "=" not in field:
            continue

        key, value = field.split("=", 1)

        key = key.strip()
        value = value.strip()

        if key.lower() == "cause":
            cause = value

        elif key.lower() == "count":
            count = value

    return {
        "timestamp": timestamp,
        "rat": rat,
        "protocol": protocol,
        "event": event,
        "ue_id": ue_id,
        "cause": cause,
        "count": count
    }
