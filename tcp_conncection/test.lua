local function load_state(i)
    local state = savestate.create(i+1)
    savestate.load(state)
end

local function draw_joypad_gui(reward, total_reward)
    local left_press = joypad.get(1)['left']
    local right_press = joypad.get(1)['right']
    local B_press = joypad.get(1)['B']
    local A_press = joypad.get(1)['A']

    -- +======================+ --
    -- ------ GUI JOYPAD ------ --
    -- +======================+ --
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
    -- +======================+ --
    -- ---- END GUI JOYPAD ---- --
    -- +======================+ --
end

-- ================= SERVER =================

-- ================= RL STATE ================
local max_x = 0
local total_reward = 0

-- ================= LOOP ====================
while true do
    local function reset()
        local state_ix = math.random(1, 5) -- savestates are usually 0-indexed in FCEUX
        load_state(state_ix)
        
        -- 1. Advance one frame to ensure memory addresses update to the new state
        emu.frameadvance() 
        
        -- 2. Read the NEW position from the save state
        local screen_num = memory.readbyte(0x006D)
        local x_in_screen = memory.readbyte(0x0086)
        local current_x = (screen_num * 256) + x_in_screen
        
        -- 3. Sync max_x so the baseline is 0 for this specific start
        max_x = current_x 
        total_reward = 0
    end


    -- -------- DATA / REWARD --------
    
    local dead = (memory.readbyte(0x00FC) == 0x01)
    local win  = (memory.readbyte(0x00FF) == 0x40)
    local done = dead or win

    local reward = -0.01 

    -- Math Fix: 0x006D is Screen Number, 0x0086 is X inside screen
    local screen_num = memory.readbyte(0x006D)
    local x_in_screen = memory.readbyte(0x0086)
    local raw_x = (screen_num * 256) + x_in_screen


    -- 2. Progress Reward (Scaled down to 0.1 to prevent "thousands")
    if raw_x > max_x then
        local bonus = (raw_x - max_x) * 0.1 -- FIX: Scaled reward
        reward = reward + bonus
        max_x = raw_x
    elseif raw_x < max_x then
        reward = reward - 0.02 
    end

    if dead then reward = reward - 15.0 end -- Scaled penalty
    if win then reward = reward + 50.0 end

    total_reward = total_reward + reward
    draw_joypad_gui(reward, total_reward)

    emu.frameadvance()
    if done then reset() end

end