import socket
import numpy as np
import time
import subprocess
from pathlib import Path
import cv2

fceux_dir = Path(__file__).parent.parent.parent
fceux_path = fceux_dir/ "fceux.exe"
rom_path = fceux_dir / "games" / "SuperMarioBros.zip"
lua_path = Path(__file__).parent / "tcp_server.lua"


class FceuxEnv:
    """
    This class provides an interface for communication between Python and Fceux's Lua frontend.
    
    The interface automatically launches Fceux emulator, loads the specified ROM game,
    and executes the companion Lua script for communication.
    
    Make sure the following global variables are configured above:
        fceux_path (str): Path to the Fceux executable
        rom_path (str): Path to the NES ROM file (.nes, .zip)
        lua_path (str): Path to the companion Lua script
    
    Main methods:
        reset():
            Reloads savestate #1 to return the emulator to a known initial state.
            
            Returns:
                state: A frame representing the current game screen after reset
                
        step(action):
            Sends the specified action to Lua and returns the resulting percepts.
            
            Args:
                action (numpy.array): An array containing states of all joypad buttons (either 1 or 0)
                    Convention: [LEFT, RIGHT, B, A]
                    
            Returns:
                tuple: A tuple containing:
                    - frame: The current game frame after executing the action
                    - reward: The reward value resulting from the action
                    - terminated: Boolean indicating if the episode has ended
    """

    # signals for synchronized communication
    RESET = b'reset\n' # self-explanatory
    ACTION = b"action\n" # about to send an action
    FRAME = b'frame\n' # ready to process frame 
    DONE = b'done\n' # finished loading the frame, other operations can be carried out
    DATA = b'data\n'# request any other data (reward, internal game states etc.)


    def __init__(self, host="127.0.0.1", port=5000, width=256, height=240):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.frame_size = width * height * 4  # ARGB
        self.socket = None
        self.connection = None
        self.process = None # fceux process

        # start the emulator
        self._start_fceux()
        self._connect_to_server()


    def _start_fceux(self):
        """Start FCEUX with the Lua script and ROM"""
        print(">> Starting FCEUX with Lua script...")
        self.process = subprocess.Popen([fceux_path, '-lua', lua_path, rom_path])
        time.sleep(0.5)  # Give FCEUX time to start


    def _connect_to_server(self):
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


    def _preprocess(self, frame):
        # 1. Crop HUD
        frame = frame[64:, :, :]  # (208 - 32, 256, 3) 

        # Manual weighted grayscale to boost red vs green
        r = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        b = frame[:, :, 2].astype(np.float32)
        frame = 0.299*r+0.587*g+0.114*b

        # 2. Resize to preserve
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # 3. Convert to float
        # frame = frame.astype(np.float32) / 255.0
        frame = np.round(frame).astype(np.uint8)

        return frame
    

    def reset(self):
        self.socket.sendall(self.RESET)
        state, _, _ = self.step()

        return state


    def step(self, action=None):
        # 1. Send action to Lua
        if action is not None:
            self.socket.sendall(self.ACTION)
            self.socket.sendall(action.tobytes())

        # 2. Request frame
        self.socket.sendall(self.FRAME)
        frame = self._receive_frame()
        frame = self._preprocess(frame)
        self.socket.sendall(self.DONE)

        # 3. Request reward + terminated
        self.socket.sendall(self.DATA)
        data = self.socket.recv(32).strip().decode()
        reward, terminated = data.split(':')
        reward = float(reward)
        terminated = (terminated=='1')
        
        return frame, reward, terminated


    def close(self):
        """Close the connection."""
        print(">> Closing connection...")
        if self.socket:
            self.socket.close()
            self.socket = None
        self.process.terminate()
        print(">> Connection closed!")