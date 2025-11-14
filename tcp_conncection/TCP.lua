package.path = package.path .. ";./LuaSocket/?.lua"
package.cpath = package.cpath .. ";./LuaSocket/?.dll"

local socket = require("socket")
local tcp = assert(socket.tcp())
tcp:connect("127.0.0.1", 5000)
tcp:settimeout(1)

-- NES screen size
local width = 256
local height = 240

function send_frame(count)
    local img_str = gui.gdscreenshot(true)
    local raw = img_str:sub(12)


    -- -- For debugging frames; overwrites Alpha (first byte) with frame number
    -- local new_raw = {}
    -- local frame_byte = string.char(count % 256)
    -- for i = 1, #raw, 4 do
    --     new_raw[#new_raw + 1] = frame_byte .. raw:sub(i + 1, i + 3) -- R,G,B remain
    -- end
    -- tcp:send(table.concat(new_raw))

    tcp:send(raw)
end

local count = 0
while true do
    count = count + 1

    local signal, err = tcp:receive("*l") -- read ok
    if signal == "ok" then
        send_frame(count)
        emu.frameadvance()
    end
end
