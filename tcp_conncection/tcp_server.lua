package.path = package.path .. ";./LuaSocket/?.lua"
package.cpath = package.cpath .. ";./LuaSocket/?.dll"
local socket = require("socket")

-- ================== UTIL ==================
local function clear_input()
    joypad.set(1, {left=false, right=false, A=false, B=false})
end

local function send_frame(client)
    local img_str = gui.gdscreenshot(true)
    local raw = img_str:sub(12)
    client:send(raw)
end

local function load_state(i)
    local state = savestate.create(i+1)
    savestate.load(state)
end

local function draw_joypad_gui(reward, total_reward)
    local left_press = joypad.get(1)['left']
    local right_press = joypad.get(1)['right']
    local B_press = joypad.get(1)['B']
    local A_press = joypad.get(1)['A']
    local startX = 10
    local startY = 10
    local boxSize = 12

    -- Helper function to draw a "Button"
    local function drawButton(x, y, label, isPressed, activeColor)
        local bgColor = isPressed and activeColor or "black"
        local textColor = isPressed and "white" or "gray"
        
        -- Draw the button background
        gui.drawbox(x, y, x + boxSize, y + boxSize, bgColor, "white")
        -- Center the label (offset by ~3 pixels)
        gui.text(x + 3, y + 2, label, textColor)
    end

    -- Draw the Dashboard Background
    gui.drawbox(startX - 5, startY - 5, startX + 100, startY + 45, "#00000080", "white")

    -- Draw the Buttons
    drawButton(startX,      startY + 15, "<-", left_press,  "red")
    drawButton(startX + 15, startY + 15, "->", right_press, "red")
    drawButton(startX + 40, startY + 15, "B", B_press,     "yellow")
    drawButton(startX + 55, startY + 15, "A", A_press,     "green")

    -- Draw Reward Info next to it
    gui.text(startX, startY, string.format("Reward: %.2f", reward), "white")
    gui.text(startX, startY + 30, string.format("Total : %.2f", total_reward), "cyan")
end

-- ================= SERVER =================
local server = assert(socket.bind("127.0.0.1", 5000))
print(">> Lua TCP server listening on 127.0.0.1:5000")

print(">> Waiting for Python connection...")
local client = server:accept()
client:settimeout(1)
print(">> Python connected!")
emu.speedmode("maximum")

-- ================= GLOBAL STATES ================
local max_x = 0
local total_reward = 0
local current_reward = 0
local current_input = {} -- cache input too keep it persistent during frameskips
local current_done = false -- cache the current DONE state for reset logic

-- ================= MAIN LOOP ====================
while true do
    local signal = client:receive("*l")
    if not signal then break end

    -- -------- ACTION --------
    if signal == "action" then
        local bool_data = client:receive(4)
        if not bool_data then break end
        
        current_input = {
            left  = string.byte(bool_data, 1) == 1,
            right = string.byte(bool_data, 2) == 1,
            B     = string.byte(bool_data, 3) == 1,
            A     = string.byte(bool_data, 4) == 1
        }

        joypad.set(1, current_input)

    -- -------- RESET --------
    elseif signal == "reset" then
        local state_ix = math.random(1, 5)
        -- state_ix = 1
        load_state(state_ix)

        -- Clear the input state as well as the cached input from previous states
        clear_input()
        current_input = {left=false, right=false, A=false, B=false}

        -- Advance a few states to stabilize and also skip the loadstate text bubble
        for i=1,300 do
            clear_input()
            emu.frameadvance()
        end
        
        -- Read the new position from the save state
        local screen_num = memory.readbyte(0x006D)
        local x_in_screen = memory.readbyte(0x0086)
        local current_x = (screen_num * 256) + x_in_screen
        
        -- Sync max_x so the baseline is 0 for this specific start
        max_x = current_x 
        total_reward = 0
        current_done = false

    -- -------- FRAME --------
    elseif signal == "frame" then
        send_frame(client)
        for i=1, 4 do -- Repeat the action for 6 frames
            if current_done then -- if died during these frames, disable input and quit
                break
            end
            emu.frameadvance()
            joypad.set(1, current_input) --keep input constant when skipping frames
            draw_joypad_gui(current_reward, total_reward)
            current_done = (memory.readbyte(0x00FC) == 0x01) or (memory.readbyte(0x00FF) == 0x40)
        end
        client:receive("*l") -- Wait for Python ack
        if current_done then signal = "data" end -- if died flag is on, skip to the data routine to trigger a reset

    -- -------- DATA / REWARD --------
    elseif signal == "data" then
        local dead = (memory.readbyte(0x00FC) == 0x01)
        local win  = (memory.readbyte(0x00FF) == 0x40)
        local done = dead or win

        local reward = -0.1 

        local screen_num = memory.readbyte(0x006D)
        local x_in_screen = memory.readbyte(0x0086)
        local raw_x = (screen_num * 256) + x_in_screen -- correct formula for fceux!

        -- Progress Reward
        if raw_x > max_x then
            local bonus = (raw_x - max_x) * 0.2 --forward progress
            reward = reward + bonus
            max_x = raw_x
        elseif raw_x < max_x then
            reward = reward - 0.5
        end

        if dead then reward = reward - 4.0 end
        if win then reward = reward + 50.0 end

        total_reward = total_reward + reward
        current_reward = reward

        -- Send result back to Python
        client:send(tostring(reward) .. ":" .. (done and "1" or "0") .. "\n")

    else
        break
    end
end

-- ================= CLEANUP ==================
client:close()
server:close()
print(">> TCP server closed")
