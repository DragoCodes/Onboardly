import os
import subprocess

import streamlit as st

"""Module for launching Streamlit applications in new terminal windows."""


def run_command_in_new_terminal(command: str) -> None:
    """Execute a command in a new terminal window.

    Args:
        command: The command string to execute.

    Raises:
        subprocess.SubprocessError: If the subprocess fails to launch.
        FileNotFoundError: If the terminal command is not found.

    """
    try:
        if os.name == "nt":  # For Windows
            subprocess.Popen(  # noqa: S602
                ["start", "cmd", "/k", command],
                shell=True,
            )
        else:  # For Linux and MacOS
            subprocess.Popen(  # noqa: S603
                ["gnome-terminal", "--", "bash", "-c", command],
            )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        st.error(f"Error running command: {e}")


# Streamlit app
st.title("OnBoardly")

st.image(
    "https://quantumshifting.wordpress.com/wp-content/uploads/2012/08/red-pill-blue-pill.jpg?w=640",
    use_container_width=True,
)

# Create two columns for the buttons
col1, col2 = st.columns(2)

# Button for FastAPI
with col1:
    if st.button("FastAPI", use_container_width=True):
        run_command_in_new_terminal(
            "uv run streamlit run website.py --server.port 8503",
        )

# Button for BentoML
with col2:
    if st.button("BentoML", use_container_width=True):
        run_command_in_new_terminal(
            "uv run streamlit run benfront.py --server.port 8504",
        )
