"""Library selection shared by Table 6 build and measurement adapters."""
import os

LIBRARIES = ("cufftdx", "tcfft", "flashfftconv", "thunderfft")


def libraries(value=None):
    value = os.environ.get("THUNDERFFT_TABLE6_LIBRARIES", "all") if value is None else value
    selected = tuple(item.strip() for item in value.split(","))
    if selected == ("all",):
        return LIBRARIES
    if not selected or any(item not in LIBRARIES for item in selected) or len(set(selected)) != len(selected):
        raise ValueError("THUNDERFFT_TABLE6_LIBRARIES must be all or unique comma-separated names: " + ",".join(LIBRARIES))
    return selected


if __name__ == "__main__":
    try:
        print("\n".join(libraries()))
    except ValueError as error:
        raise SystemExit(str(error))
