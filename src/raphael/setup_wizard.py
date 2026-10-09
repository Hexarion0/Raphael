"""Interactive setup and configuration wizard for RAPHAEL."""

import json
from getpass import getpass
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

from raphael.audio.tts import TextToSpeech, resolve_tts_engine
from raphael.config import AudioConfig, ProviderConfig, get_settings
from raphael.logging import get_logger
from raphael.platform import get_audio_backend

logger = get_logger("setup")


def custom_voice_asset_problems(configuration: dict[str, str]) -> list[str]:
    """Check local custom-voice prerequisites without loading models or downloading files."""
    root = Path(__file__).resolve().parents[2]
    defaults = AudioConfig()

    def local_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    problems = []
    python = local_path(configuration.get("TTS_CHATTERBOX_PYTHON", defaults.tts_chatterbox_python))
    if not python.is_file():
        problems.append(f"Turbo Python environment: {python}")
    profile_root = local_path(configuration.get("TTS_VOICE_PROFILES", defaults.tts_voice_profiles))
    manifest = profile_root / configuration["TTS_VOICE"] / "voice.json"
    try:
        profile = json.loads(manifest.read_text(encoding="utf-8"))
        if profile["engine"] != "chatterbox_turbo":
            raise ValueError("Expected a Chatterbox Turbo profile")
        for key in ("reference_audio", "reference_transcript"):
            path = local_path(profile[key])
            if not path.is_file() or not path.stat().st_size:
                problems.append(f"{key}: {path}")
    except (OSError, ValueError, KeyError, TypeError):
        problems.append(f"Missing or invalid voice profile: {manifest}")
    model = local_path(configuration.get("TTS_CHATTERBOX_MODEL", defaults.tts_chatterbox_model))
    for name in (
        "inventory.json", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
        "ve.safetensors", "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json",
    ):
        path = model / name
        if not path.is_file() or not path.stat().st_size:
            problems.append(f"Turbo model file: {path}")
    return problems


def run_setup_wizard() -> None:
    """Run interactive CLI configuration wizard for RAPHAEL."""
    print("\n" + "=" * 55)
    print("      RAPHAEL - Interactive Configuration Setup")
    print("=" * 55)
    print("This wizard will help you configure your AI assistant.\n")

    env_path = Path(".env")
    original_env = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
    original_keys = {key.upper(): key for key in original_env}
    current_env = {key.upper(): value for key, value in original_env.items() if value is not None}

    # 1. AI Provider Keys
    print("\n--- [1/4] AI Providers & API Keys ---")
    current_nim = current_env.get("NIM_API_KEY", "")
    if current_nim:
        preview = current_nim[:6] + "..." + (current_nim[-4:] if len(current_nim) > 10 else "")
        nim_prompt = f"NVIDIA NIM API Key [{preview}]: "
    else:
        nim_prompt = "NVIDIA NIM API Key: "
    nim_input = getpass(nim_prompt).strip()
    if nim_input:
        current_env["NIM_API_KEY"] = nim_input
    elif not current_nim:
        current_env["NIM_API_KEY"] = ""

    current_groq = current_env.get("GROQ_API_KEY", "")
    if current_groq:
        groq_prompt = f"Groq API Key (Optional) [{current_groq[:6]}...]: "
    else:
        groq_prompt = "Groq API Key (Optional): "
    groq_input = getpass(groq_prompt).strip()
    if groq_input:
        current_env["GROQ_API_KEY"] = groq_input

    current_or = current_env.get("OPENROUTER_API_KEY", "")
    if current_or:
        or_prompt = f"OpenRouter API Key (Optional) [{current_or[:6]}...]: "
    else:
        or_prompt = "OpenRouter API Key (Optional): "
    or_input = getpass(or_prompt).strip()
    if or_input:
        current_env["OPENROUTER_API_KEY"] = or_input

    # 2. Voice and Speech Settings
    print("\n--- [2/4] Text-to-Speech (TTS) Voice ---")
    print("  [1] RAPHAEL custom voice — Chatterbox Turbo (NVIDIA GPU required)")
    print("      Requires the separate Turbo environment, model, and private voice reference.")
    print("  [2] Default Amy — Piper (lighter, works on CPU)")
    current_voice = current_env.get("TTS_VOICE", "en_US-amy-medium")
    while True:
        voice_choice = input(
            f"\nSelect voice [1-2] (Enter to keep {current_voice}): "
        ).strip()
        if voice_choice in {"", "1", "2"}:
            break
        print("Please enter 1 for RAPHAEL or 2 for Amy.")
    if voice_choice == "1":
        current_env["TTS_VOICE"] = "raphael"
        current_env["TTS_ENGINE"] = "chatterbox_turbo"
    elif voice_choice == "2":
        current_env["TTS_VOICE"] = "en_US-amy-medium"
        current_env["TTS_ENGINE"] = "piper"
    else:
        current_env["TTS_VOICE"] = current_voice
        current_env.setdefault(
            "TTS_ENGINE",
            "chatterbox_turbo" if current_voice == "raphael" else resolve_tts_engine(current_voice),
        )

    # 3. Audio Devices
    print("\n--- [3/4] Audio Devices ---")
    backend = get_audio_backend()
    devices = backend.list_devices()
    input_devices = [d for d in devices if d.max_input_channels > 0]
    output_devices = [d for d in devices if d.max_output_channels > 0]

    print("\nAvailable Microphones:")
    for d in input_devices:
        default_marker = " (System Default)" if d.is_default_input else ""
        print(f"  [{d.index}] {d.name}{default_marker}")

    mic_choice = input(
        "\nMicrophone index or name [Enter to keep, 'default' for system default]: "
    ).strip()
    if mic_choice.lower() == "default":
        current_env.pop("AUDIO_INPUT_DEVICE", None)
    elif mic_choice:
        current_env["AUDIO_INPUT_DEVICE"] = mic_choice

    print("\nAvailable Speakers:")
    for d in output_devices:
        default_marker = " (System Default)" if d.is_default_output else ""
        print(f"  [{d.index}] {d.name}{default_marker}")

    spk_choice = input(
        "\nSpeaker index or name [Enter to keep, 'default' for system default]: "
    ).strip()
    if spk_choice.lower() == "default":
        current_env.pop("AUDIO_OUTPUT_DEVICE", None)
    elif spk_choice:
        current_env["AUDIO_OUTPUT_DEVICE"] = spk_choice

    # Preserve custom settings, comments, and quoting for unchanged values.
    print("\n--- [4/4] Saving Configuration ---")
    defaults = {
        "RAPHAEL_ENV": "development",
        "RAPHAEL_LOG_LEVEL": "INFO",
        "NIM_MODEL": ProviderConfig().nim_model,
        "OLLAMA_HOST": "http://localhost:11434",
        "WAKE_WORD": "hey raphael",
        "WAKE_THRESHOLD": "0.5",
        "WAKE_COOLDOWN": "2.0",
        "STT_MODEL": "base.en",
        "STT_DEVICE": "cpu",
        "STT_COMPUTE_TYPE": "int8",
        "TTS_SPEED": "1.0",
        "TTS_ENABLED": "true",
    }
    for key, value in defaults.items():
        current_env.setdefault(key, value)
    if not env_path.exists():
        env_path.touch(mode=0o600, exist_ok=False)
        env_path.write_text("# RAPHAEL Configuration File\n", encoding="utf-8")
    for key in original_keys.keys() - current_env.keys():
        if key in {"AUDIO_INPUT_DEVICE", "AUDIO_OUTPUT_DEVICE"}:
            unset_key(env_path, original_keys[key])
    for key, value in current_env.items():
        original_key = original_keys.get(key, key)
        if original_env.get(original_key) != value:
            set_key(env_path, original_key, value, quote_mode="always")
    env_path.chmod(0o600)
    print("✅ Configuration successfully saved to .env")

    # Clear cached settings so changes take effect immediately
    get_settings.cache_clear()

    # Pre-download selected TTS voice if not present
    print("\n--- Testing Voice Output ---")
    chosen_voice = current_env["TTS_VOICE"]
    custom_voice = current_env["TTS_ENGINE"] in {"chatterbox", "chatterbox_turbo"}
    problems = custom_voice_asset_problems(current_env) if custom_voice else []
    if problems:
        print("⚠️ Custom voice is not ready. Missing or invalid local assets:")
        for problem in problems:
            print(f"  - {problem}")
        print("Your choice is saved, but the custom voice audio test was skipped.")
        print("Restore/install these assets and rerun setup, or choose Amy to get started.")
        print("See docs/chatterbox-turbo-integration.md for the custom voice requirements.")
    else:
        print(f"Testing voice: '{chosen_voice}'...")
        tts = None
        try:
            tts = TextToSpeech(
                voice_name=chosen_voice, engine=current_env["TTS_ENGINE"], enabled=True,
            )
            success = tts.speak("Hi, I'm Raphael. Let's check that you can hear me.", block=True)
            if success:
                print("✅ Audio test successful!")
                if custom_voice:
                    print("If Turbo failed, playback used a local Piper fallback; check warnings.")
            else:
                print("⚠️ Audio test failed: speech playback did not succeed.")
        except Exception as err:
            print(f"⚠️ Audio test skipped or failed: {err}")
        finally:
            if tts is not None:
                tts.close()

    print("\n" + "=" * 55)
    if problems:
        print("  Configuration saved. Custom voice setup needs the assets listed above.")
    else:
        print("  Setup Complete! Run: raphael start")
    print("=" * 55 + "\n")
