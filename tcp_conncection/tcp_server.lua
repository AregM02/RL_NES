package.path = package.path .. ";./LuaSocket/?.lua"
package.cpath = package.cpath .. ";./LuaSocket/?.dll"


-- +========= SOME USEFUL FUNCTIONS =============+
-- capture and send a frame
function send_frame(client)
    local img_str = gui.gdscreenshot(true)
    local raw = img_str:sub(12) -- Remove header
    client:send(raw)
end

-- load level 1 (from state 1)
function load_level_1()
    local state = savestate.create(2)
    savestate.load(state)
end

-- +=============================================+


-- SERVER START
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


while true do
    local signal, _ = client:receive("*l")

    if not signal then break end

    if signal == "action" then
        local bool_data, _ = client:receive(4) -- read action (4 bytes)
        local vals = {}

        for i = 1, #bool_data do
            local byte = string.byte(bool_data, i)
            vals[i] = (byte == 1)
        end

        local input = {}
        input["left"] = vals[1]
        input["right"] = vals[2]
        input["B"] = vals[3]
        input["A"] = vals[4]
        joypad.set(1, input)
    elseif signal == "reset" then
        load_level_1()
    elseif signal == "frame" then
        send_frame(client)
        emu.frameadvance()
        _, _ = client:receive("*l") -- wait for python to read the frame
    elseif signal == "data" then
        -- game specific addresses, design as you need
        local state = memory.readbyte(0x000E) -- get player state
        local done = (state == 0x0B)          -- true if dying

        local reward = 100

        client:send(string.char(reward, done and 1 or 0))
    else
        break
    end
end

-- Cleanup
client:close()
server:close()
print(">> TCP server closed")
