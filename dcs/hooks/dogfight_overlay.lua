--[[
    Dogfight Supporter — 画面オーバーレイ表示フック

    Python 推論サーバーから UDP で受信した有利度スコアを
    DCS の画面上にリアルタイム表示する。

    スコア表示:
        +1.0 = 自機がガンキル位置（緑）
         0.0 = 互角（黄）
        -1.0 = 敵機がガンキル位置（赤）

    配置先:
      %USERPROFILE%/Saved Games/DCS.openbeta/Scripts/Hooks/dogfight_overlay.lua
]]

-- ============================================================
-- 設定
-- ============================================================
local DS_OVERLAY = {
    host = "127.0.0.1",
    port = 9090,

    display_duration = 2.0,
    font_size = 24,
    
    x = 0.02,
    y = 0.15,

    colors = {
        advantage = {         -- 有利: 緑
            r = 50, g = 255, b = 100, a = 230
        },
        neutral = {           -- 互角: 黄
            r = 255, g = 220, b = 50, a = 230
        },
        disadvantage = {      -- 不利: 赤
            r = 255, g = 70, b = 70, a = 230
        },
        background = {
            r = 0, g = 0, b = 0, a = 140
        },
        text_secondary = {
            r = 200, g = 200, b = 200, a = 200
        },
    },
}

-- ============================================================
-- 状態変数
-- ============================================================
local socket = nil
local udp_recv = nil
local current_result = nil
local last_update_time = 0

-- ============================================================
-- 初期化
-- ============================================================
local function DS_overlay_init()
    local status, sock = pcall(require, "socket")
    if not status then
        log.write("DogfightOverlay", log.WARNING, "LuaSocket not available")
        return false
    end
    socket = sock

    udp_recv = socket.udp()
    if not udp_recv then
        log.write("DogfightOverlay", log.ERROR, "Failed to create UDP socket")
        return false
    end

    udp_recv:settimeout(0)
    local ok, err = udp_recv:setsockname(DS_OVERLAY.host, DS_OVERLAY.port)
    if not ok then
        log.write("DogfightOverlay", log.ERROR, "Bind failed: " .. tostring(err))
        return false
    end

    log.write("DogfightOverlay", log.INFO, 
        "Listening on " .. DS_OVERLAY.host .. ":" .. DS_OVERLAY.port)
    return true
end

-- ============================================================
-- JSON パーサー（軽量版）
-- ============================================================
local function parse_json_simple(str)
    local result = {}

    -- score (float, 正負対応)
    local score = str:match('"score"%s*:%s*([%-]?[%d%.]+)')
    if score then result.score = tonumber(score) end

    -- category
    local cat = str:match('"category"%s*:%s*"([^"]*)"')
    if cat then result.category = cat end

    -- category_ja
    local cat_ja = str:match('"category_ja"%s*:%s*"([^"]*)"')
    if cat_ja then result.category_ja = cat_ja end

    -- timestamp
    local ts = str:match('"timestamp"%s*:%s*([%d%.]+)')
    if ts then result.timestamp = tonumber(ts) end

    return result
end

-- ============================================================
-- スコアに応じた色を補間
-- ============================================================
local function score_to_color(score)
    -- score: -1 ~ +1
    -- -1 → 赤, 0 → 黄, +1 → 緑
    local c = DS_OVERLAY.colors
    
    if score >= 0 then
        -- 0 → 黄, 1 → 緑
        local t = math.min(score, 1.0)
        return {
            r = math.floor(c.neutral.r + (c.advantage.r - c.neutral.r) * t),
            g = math.floor(c.neutral.g + (c.advantage.g - c.neutral.g) * t),
            b = math.floor(c.neutral.b + (c.advantage.b - c.neutral.b) * t),
            a = 230,
        }
    else
        -- -1 → 赤, 0 → 黄
        local t = math.min(-score, 1.0)
        return {
            r = math.floor(c.neutral.r + (c.disadvantage.r - c.neutral.r) * t),
            g = math.floor(c.neutral.g + (c.disadvantage.g - c.neutral.g) * t),
            b = math.floor(c.neutral.b + (c.disadvantage.b - c.neutral.b) * t),
            a = 230,
        }
    end
end

-- ============================================================
-- データ受信
-- ============================================================
local function DS_receive_result()
    if not udp_recv then return end

    local latest = nil
    while true do
        local data = udp_recv:receive()
        if not data then break end
        latest = data
    end

    if latest then
        local parsed = parse_json_simple(latest)
        if parsed and parsed.score then
            current_result = parsed
            last_update_time = DCS.getRealTime()
        end
    end
end

-- ============================================================
-- 画面描画
-- ============================================================
local function DS_draw_overlay()
    if not current_result then return end

    local now = DCS.getRealTime()
    if now - last_update_time > DS_OVERLAY.display_duration then
        return
    end

    local score = current_result.score or 0
    local category_ja = current_result.category_ja or "---"

    -- スコアに応じた色を取得
    local color = score_to_color(score)
    local bg = DS_OVERLAY.colors.background
    local txt2 = DS_OVERLAY.colors.text_secondary

    -- 画面サイズ取得
    local w, h = DCS.getScreenSize()
    local px = DS_OVERLAY.x * w
    local py = DS_OVERLAY.y * h

    local box_w = 300
    local box_h = 90

    -- 背景描画
    DCS.setDrawColor(bg.r, bg.g, bg.b, bg.a)
    DCS.fillRect(px, py, box_w, box_h)

    -- 枠線（色付き）
    DCS.setDrawColor(color.r, color.g, color.b, color.a)
    DCS.drawRect(px, py, box_w, box_h)
    DCS.fillRect(px, py, box_w, 3)

    -- カテゴリ表示
    DCS.setDrawColor(color.r, color.g, color.b, color.a)
    DCS.drawText(
        px + 10,
        py + 8,
        DS_OVERLAY.font_size,
        category_ja
    )

    -- スコア数値
    DCS.drawText(
        px + 180,
        py + 8,
        DS_OVERLAY.font_size,
        string.format("%+.2f", score)
    )

    -- スコアバー (中央が 0, 左が -1, 右が +1)
    local bar_x = px + 10
    local bar_y = py + 50
    local bar_w = box_w - 20
    local bar_h = 10
    local center_x = bar_x + bar_w / 2

    -- バー背景
    DCS.setDrawColor(60, 60, 60, 180)
    DCS.fillRect(bar_x, bar_y, bar_w, bar_h)

    -- 中央マーカー
    DCS.setDrawColor(150, 150, 150, 200)
    DCS.fillRect(center_x - 1, bar_y - 2, 2, bar_h + 4)

    -- スコアインジケータ
    DCS.setDrawColor(color.r, color.g, color.b, color.a)
    if score >= 0 then
        local fill_w = (score * bar_w / 2)
        DCS.fillRect(center_x, bar_y, fill_w, bar_h)
    else
        local fill_w = (-score * bar_w / 2)
        DCS.fillRect(center_x - fill_w, bar_y, fill_w, bar_h)
    end

    -- ラベル
    DCS.setDrawColor(txt2.r, txt2.g, txt2.b, txt2.a)
    DCS.drawText(bar_x, py + 68, 12, "DISADV")
    DCS.drawText(bar_x + bar_w - 28, py + 68, 12, "ADV")
end

-- ============================================================
-- DCS Hook コールバック登録
-- ============================================================
local initialized = false

local ds_callbacks = {}

function ds_callbacks.onSimulationFrame()
    if not initialized then
        initialized = DS_overlay_init()
    end

    if initialized then
        DS_receive_result()
        DS_draw_overlay()
    end
end

function ds_callbacks.onSimulationStop()
    if udp_recv then
        udp_recv:close()
        udp_recv = nil
    end
    initialized = false
    current_result = nil
    log.write("DogfightOverlay", log.INFO, "Overlay stopped")
end

DCS.setUserCallbacks(ds_callbacks)

log.write("DogfightOverlay", log.INFO, "Dogfight Supporter Overlay Hook loaded")
