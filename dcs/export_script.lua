--[[
    Dogfight Supporter — DCS Export Script
    
    二機の戦闘機のデータを収集し、UDP でPython サーバーに送信する。
    
    配置先:
      %USERPROFILE%/Saved Games/DCS.openbeta/Scripts/Export.lua
      (既存の Export.lua がある場合は dofile() でインクルード)
    
    Usage:
      既存の Export.lua に以下を追加:
        dofile(lfs.writedir() .. "Scripts/DogfightSupporter/export_script.lua")
      
      または本ファイルを直接 Export.lua としてコピー
]]

-- ============================================================
-- 設定
-- ============================================================
local DS_CONFIG = {
    host = "127.0.0.1",
    port = 9089,
    update_interval = 1 / 30,  -- 30Hz でデータ送信
    enabled = true,
    debug = false,

    -- マルチプレイ設定
    -- サーバースクリプトから敵データを受信する場合のポート
    -- (サーバー側にスクリプトが設置されている場合のみ有効)
    server_data_port = 9091,
    use_server_data = false,   -- true: サーバーからの敵データを使用
}

-- ============================================================
-- UDP ソケット初期化
-- ============================================================
local socket = require("socket")
local udp_socket = nil
local last_update_time = 0

local function DS_init()
    udp_socket = socket.udp()
    if udp_socket then
        udp_socket:settimeout(0)
        udp_socket:setpeername(DS_CONFIG.host, DS_CONFIG.port)
        if DS_CONFIG.debug then
            log.write("DogfightSupporter", log.INFO, "UDP socket initialized -> "
                .. DS_CONFIG.host .. ":" .. DS_CONFIG.port)
        end
    end
end

-- ============================================================
-- ユーティリティ関数
-- ============================================================

--- テーブルを JSON 文字列にシリアライズ（軽量版）
local function to_json(tbl)
    local parts = {}
    for k, v in pairs(tbl) do
        local val_str
        if type(v) == "table" then
            val_str = to_json(v)
        elseif type(v) == "string" then
            val_str = '"' .. v .. '"'
        elseif type(v) == "number" then
            val_str = string.format("%.6f", v)
        elseif type(v) == "boolean" then
            val_str = tostring(v)
        else
            val_str = '"' .. tostring(v) .. '"'
        end
        parts[#parts + 1] = '"' .. k .. '":' .. val_str
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

--- Position マトリクスからヘディング・ピッチ・バンクを算出
--- DCS の Position は 3x3 回転行列 + 位置ベクトル
local function extract_euler_angles(pos)
    if not pos or not pos.x then
        return 0, 0, 0
    end
    
    -- ヘディング (yaw): X軸（前方ベクトル）の水平成分から算出
    local heading = math.atan2(pos.x.z, pos.x.x)
    
    -- ピッチ: 前方ベクトルのY成分
    local pitch = math.asin(pos.x.y)
    
    -- バンク (roll): Y軸（上方ベクトル）から算出
    local bank = math.atan2(-pos.z.y, pos.y.y)
    
    return heading, pitch, bank
end

--- 3Dベクトルの大きさ
local function vec_magnitude(x, y, z)
    return math.sqrt(x * x + y * y + z * z)
end

-- ============================================================
-- 自機データ取得
-- ============================================================
local function get_self_data()
    local data = {}

    -- 基本データ
    local self_data = LoGetSelfData()
    if not self_data then return nil end

    data.name = self_data.Name or "Unknown"
    data.type = self_data.Type or "Unknown"

    -- 位置 (ワールド座標系)
    if self_data.Position then
        data.x = self_data.Position.p.x
        data.y = self_data.Position.p.y  -- 高度
        data.z = self_data.Position.p.z
        
        -- 姿勢角
        data.heading, data.pitch, data.bank = extract_euler_angles(self_data.Position)
    end

    -- 速度
    data.TAS = LoGetTrueAirSpeed() or 0        -- True Airspeed (m/s)
    data.IAS = LoGetIndicatedAirSpeed() or 0    -- Indicated Airspeed (m/s)
    data.mach = LoGetMachNumber() or 0          -- マッハ数

    -- 高度
    data.alt_asl = LoGetAltitudeAboveSeaLevel() or 0     -- 海面高度 (m)
    data.alt_agl = LoGetAltitudeAboveGroundLevel() or 0  -- 対地高度 (m)

    -- 迎角・横滑り角
    data.AoA = LoGetAngleOfAttack() or 0       -- 迎角 (rad)
    data.AoS = LoGetSideDeviation() or 0       -- 横滑り角 (rad)

    -- 加速度 (G)
    local acc = LoGetAccelerationUnits()
    if acc then
        data.Gx = acc.x or 0
        data.Gy = acc.y or 0  -- 垂直荷重 (Ny)
        data.Gz = acc.z or 0
    else
        data.Gx, data.Gy, data.Gz = 0, 0, 0
    end

    -- 角速度 (rad/s)
    local ang_vel = LoGetAngularVelocity()
    if ang_vel then
        data.roll_rate = ang_vel.x or 0
        data.pitch_rate = ang_vel.y or 0
        data.yaw_rate = ang_vel.z or 0
    else
        data.roll_rate, data.pitch_rate, data.yaw_rate = 0, 0, 0
    end

    -- エンジン情報
    local eng = LoGetEngineInfo()
    if eng and eng.RPM then
        data.throttle = eng.RPM.left or 0
        data.fuel_internal = eng.fuel_internal or 0
    else
        data.throttle = 0
        data.fuel_internal = 0
    end

    return data
end

-- ============================================================
-- 敵機データ取得 (LoGetWorldObjects)
-- ============================================================
local function get_enemy_data()
    local objects = LoGetWorldObjects("units")
    if not objects then return nil end
    
    local self_data = LoGetSelfData()
    if not self_data then return nil end
    
    local self_coalition = self_data.Coalition
    local enemies = {}
    
    for id, obj in pairs(objects) do
        -- 航空機のみ、かつ敵陣営のみ
        if obj.Type and obj.Type.level1 == 1 then  -- level1 = 1: Aircraft
            if obj.Coalition and obj.Coalition ~= self_coalition then
                local enemy = {}
                enemy.id = id
                enemy.name = obj.Name or "Unknown"
                enemy.type = obj.Type or {}
                
                -- 位置
                if obj.LatLongAlt then
                    enemy.lat = obj.LatLongAlt.Lat or 0
                    enemy.lon = obj.LatLongAlt.Long or 0
                    enemy.alt = obj.LatLongAlt.Alt or 0
                end
                
                if obj.Position then
                    enemy.x = obj.Position.p.x
                    enemy.y = obj.Position.p.y
                    enemy.z = obj.Position.p.z
                    
                    enemy.heading, enemy.pitch, enemy.bank = extract_euler_angles(obj.Position)
                end
                
                -- 速度
                if obj.Position and obj.Position.x then
                    -- 速度ベクトルは直接取得できないため、フレーム間差分で推定
                    -- ここでは方向ベクトルのみ記録
                    enemy.forward_x = obj.Position.x.x
                    enemy.forward_y = obj.Position.x.y
                    enemy.forward_z = obj.Position.x.z
                end
                
                enemies[#enemies + 1] = enemy
            end
        end
    end
    
    -- 最も近い敵機を返す（複数敵がいる場合）
    if #enemies == 0 then return nil end
    
    if #enemies == 1 then return enemies[1] end
    
    -- 距離でソート
    local self_x = self_data.Position.p.x
    local self_y = self_data.Position.p.y
    local self_z = self_data.Position.p.z
    
    table.sort(enemies, function(a, b)
        local da = vec_magnitude(a.x - self_x, a.y - self_y, a.z - self_z)
        local db = vec_magnitude(b.x - self_x, b.y - self_y, b.z - self_z)
        return da < db
    end)
    
    return enemies[1]
end

-- ============================================================
-- メインエクスポートループ
-- ============================================================
local function DS_export_frame()
    if not DS_CONFIG.enabled then return end
    if not udp_socket then return end
    
    local current_time = LoGetModelTime()
    if not current_time then return end
    
    -- 更新レート制御
    if current_time - last_update_time < DS_CONFIG.update_interval then
        return
    end
    last_update_time = current_time
    
    -- 自機データ
    local self_data = get_self_data()
    if not self_data then return end
    
    -- 敵機データ（マルチプレイではnilになる可能性あり）
    local enemy_data = get_enemy_data()
    
    -- パケット構築
    local packet = {
        timestamp = current_time,
        model_time = current_time,
        self_aircraft = self_data,
    }
    
    if enemy_data then
        packet.enemy_aircraft = enemy_data
        packet.has_enemy = true
        packet.self_only = false
    else
        packet.has_enemy = false
        packet.self_only = true  -- Python側で自機のみモードを使用
    end
    
    -- JSON シリアライズ & 送信
    -- 敵データがなくても自機データは常に送信（マルチプレイ対応）
    local json_str = to_json(packet)
    udp_socket:send(json_str)
    
    if DS_CONFIG.debug then
        log.write("DogfightSupporter", log.INFO, "Sent: " .. string.sub(json_str, 1, 200))
    end
end

-- ============================================================
-- DCS Export コールバック登録
-- ============================================================

-- 既存の Export 関数をチェーン
local prev_LuaExportStart = LuaExportStart
local prev_LuaExportBeforeNextFrame = LuaExportBeforeNextFrame
local prev_LuaExportStop = LuaExportStop

LuaExportStart = function()
    DS_init()
    if prev_LuaExportStart then
        prev_LuaExportStart()
    end
end

LuaExportBeforeNextFrame = function()
    DS_export_frame()
    if prev_LuaExportBeforeNextFrame then
        prev_LuaExportBeforeNextFrame()
    end
end

LuaExportStop = function()
    if udp_socket then
        udp_socket:close()
        udp_socket = nil
    end
    if prev_LuaExportStop then
        prev_LuaExportStop()
    end
end
