package.path = package.path .. ";./LuaSocket/?.lua"
package.cpath = package.cpath .. ";./LuaSocket/?.dll"

local socket = require("socket")

-- NES screen size
local width = 256
local height = 240

-- Create TCP server
local server = assert(socket.bind("127.0.0.1", 5000))
print(">> Lua TCP server listening on 127.0.0.1:5000")

-- Accept a client (Python)
print(">> Waiting for Python connection...")
local client = server:accept()
client:settimeout(1)
print(">> Python connected!")

-- Function to capture and send a frame
function send_frame()
    local img_str = gui.gdscreenshot(true)
    local raw = img_str:sub(12) -- Remove header
    client:send(raw)
end

-- Startup: Run 10 frames to initialize the game state
print(">> Running startup: 20-frames + load state")
for i = 1, 20 do
    emu.frameadvance()
end
print(">> Startup initialization complete!")


while true do
    -- Wait for "ok" from Python
    local signal, err = client:receive("*l")
    if signal == "ok" then
        send_frame()
        emu.frameadvance()
    else
        do
            break
        end
    end
end

-- Cleanup
client:close()
server:close()
print(">> TCP server closed")
