import socket
import numpy as np
import time
import subprocess
from pathlib import Path

fceux_path = Path(__file__).parent.parent.parent / "fceux64.exe"
rom_path = Path(__file__).parent.parent.parent / "games" / "SuperMarioBros.zip"
lua_path = Path(__file__).parent / "tcp_server.lua"


class FceuxEnv:
    """
    TCP interface to FCEUX to retrieve percepts (only RGB screen frames for now (HxWx3 uint8)).
    """

    OK_MESSAGE = b'ok\n'

    def __init__(self, host="127.0.0.1", port=5000, width=256, height=240):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.frame_size = width * height * 4  # ARGB
        self.socket = None
        self.connection = None
        self.process = None # fceux process

        self.start_fceux()
        self.connect_to_server()

    def start_fceux(self):
        """Start FCEUX with the Lua script and ROM"""
        print(">> Starting FCEUX with Lua script...")
        self.process = subprocess.Popen([fceux_path, '-lua', lua_path, rom_path])
        time.sleep(2)  # Give FCEUX time to start

    def connect_to_server(self):
        """Connect to the Lua TCP server"""
        print(f">> Connecting to Lua TCP server at {self.host}:{self.port}...")
        
        start_time = time.time()
        timeout = 15
        
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(">> Connected to Lua TCP server!")
                break
            except (ConnectionRefusedError, socket.timeout):
                if time.time() - start_time > timeout:
                    raise RuntimeError("Timeout: could not connect to Lua TCP server")
                time.sleep(0.5)
            except Exception as e:
                print(f"Connection error: {e}")
                if time.time() - start_time > timeout:
                    raise RuntimeError("Timeout: could not connect to Lua TCP server")
                time.sleep(0.5)

    def _receive_frame(self):
        """Receive exactly one ARGB frame from the server."""
        buf = bytearray(self.frame_size)
        mv = memoryview(buf)
        received = 0
        while received < self.frame_size:
            try:
                n = self.socket.recv_into(mv[received:], self.frame_size - received)
                if n == 0:
                    return None
                received += n
            except:
                print(">> Socket ended by FCEUX during frame reception...")
                # self.close()
                return None
                
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 4))
        rgb = arr[:, :, 1:4]  # strip alpha
        return rgb

    def step(self):
        """
        Request the next frame from the server and return RGB array.
        Returns None if the connection is closed.
        """
        try:
            self.socket.sendall(self.OK_MESSAGE)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print(">> Lua/FCEUX closed connection while sending ACK.")
            self.close()
            return None

        frame = self._receive_frame()
        return frame

    def close(self):
        """Close the connection."""
        print(">> Closing connection...")
        if self.socket:
            self.socket.close()
            self.socket = None
        self.process.terminate()
        print(">> Connection closed!")