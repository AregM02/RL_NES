import socket
import numpy as np
import time


class FceuxEnv:
    """
    TCP interface to FCEUX to retrieve RGB screen frames (HxWx3 uint8).
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

        self.start_server()

    def start_server(self):
        """Start the TCP server and wait for a client connection."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f">> Waiting for TCP connection on {self.host}:{self.port}...")
        self.connection, _ = self.socket.accept()
        print(">> Connected! Retrieving frame data...")

    def _receive_frame(self):
        """Receive exactly one ARGB frame from the client."""
        buf = bytearray(self.frame_size)
        mv = memoryview(buf)
        received = 0
        while received < self.frame_size:
            try:
                n = self.connection.recv_into(mv[received:], self.frame_size - received)
                if n == 0:
                    return None
                received += n

            except:
                print(">> Socket ended by FCEUX during frame reception...")
                self.close()
                return None
                
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 4))
        rgb = arr[:, :, 1:4]  # strip alpha
        return rgb

    def step(self):
        """
        Request the next frame from the client and return RGB array.
        Returns None if the connection is closed.
        """
        if self.connection is None:
            raise RuntimeError(">> Connection not established. Call start_server() first.")

        try:
            self.connection.sendall(self.OK_MESSAGE)
        except (ConnectionResetError, ConnectionAbortedError):
            print(">> Lua/FCEUX closed connection while sending ACK.")
            self.close()
            return None

        frame = self._receive_frame()

        return frame

    def close(self):
        """Close the connection and server socket."""
        print(">> Closing server...")
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.socket:
            self.socket.close()
            self.socket = None
        print(">> Server closed!")

