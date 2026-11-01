"""
audio_devices.py — input-device ranking shared by stt-server.py and the control panel.

Pure: takes device dicts, opens nothing.
"""


def rank_mic_candidates(devices: list[dict], keyword: str, preferred_rate: int) -> list[tuple]:
    """
    devices: [{"index", "name", "channels", "rate"}] input devices (channels > 0).
    Return keyword matches as (index, rate, channels, name), sorted by preference:
      1. starts-with-"Microphone" + preferred_rate  (clean_match)
      2. preferred_rate                              (preferred_match)
      3. starts-with-"Microphone"                   (plain_match)
      4. any match                                   (first_match)
    """
    clean, preferred, plain, other = [], [], [], []
    for d in devices:
        if keyword.lower() not in d["name"].lower():
            continue
        is_plain = d["name"].lower().startswith("microphone")
        entry = (d["index"], d["rate"], d["channels"], d["name"])
        if is_plain and d["rate"] == preferred_rate:
            clean.append(entry)
        elif d["rate"] == preferred_rate:
            preferred.append(entry)
        elif is_plain:
            plain.append(entry)
        else:
            other.append(entry)
    return clean + preferred + plain + other


def list_input_devices(p) -> list[dict]:
    """Input devices from an open pyaudio.PyAudio(), with host API names."""
    out = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        ch = info.get("maxInputChannels", 0)
        if ch <= 0:
            continue
        out.append({"index": i, "name": info.get("name", ""), "channels": ch,
                    "rate": int(info["defaultSampleRate"]),
                    "host_api": p.get_host_api_info_by_index(info["hostApi"])["name"]})
    return out
