#!/usr/bin/env python3
"""
Launcher script for the ARC Challenge Streamlit app
"""

import subprocess
import sys
import os


def main():
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check if .env file exists
    if not os.path.exists(".env"):
        print(
            "⚠️  Warning: No .env file found. You may need to create one with your OPENROUTER_API_KEY"
        )
        print("   Example: echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env")
        print()

    # Run streamlit
    try:
        print("🚀 Starting ARC Challenge Solver...")
        print("🌐 The app will open in your browser shortly")
        print("🛑 Press Ctrl+C to stop the server")
        print()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
                "--browser.serverAddress",
                "localhost",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down ARC Challenge Solver")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
