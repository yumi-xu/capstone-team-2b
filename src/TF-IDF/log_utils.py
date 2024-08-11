from datetime import datetime


def format_timedelta(td):
    days = td.days
    seconds = td.seconds
    microseconds = td.microseconds

    # Convert total seconds to hours, minutes, and remaining seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create the formatted string
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    # Join the parts with commas and return the result
    return ", ".join(parts) if parts else "0 seconds"


_last_log_time = datetime.now()


def log(*args):
    time = str(datetime.now())
    print(f"[{time}]", *args)


def log_start(*args):
    global _last_log_time
    _last_log_time = datetime.now()
    print(f"[{_last_log_time}]", *args)


def log_end(*args):
    global _last_log_time
    before = _last_log_time
    now = datetime.now()
    _last_log_time = now
    td = now - before
    print(f"[{_last_log_time}]", *args, f"(cost {format_timedelta(td)})")
