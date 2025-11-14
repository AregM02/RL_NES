from tcp_conncection.environment import FceuxEnv
import time

env = FceuxEnv()

while True:
    frame = env.step()
    if frame is None:
        break
    
    time.sleep(0.05)
    print(frame.shape)