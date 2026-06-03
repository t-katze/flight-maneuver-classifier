local Tacview = require("Tacview180")

local ADDON_NAME = "Flight Maneuver Manual Label Editor"
local ADDON_VERSION = "1.0.0"

Tacview.AddOns.Current.SetTitle(ADDON_NAME)
Tacview.AddOns.Current.SetVersion(ADDON_VERSION)
Tacview.AddOns.Current.SetAuthor("OpenAI")
Tacview.AddOns.Current.SetNotes("Right-click a fixed-wing aircraft in the 3D view to assign a manual maneuver label and export edits to CSV.")

local MAIN_LABELS =
{
	"Straight_Level",
	"Straight_Climb",
	"Straight_Descent",
	"Level_Turn",
	"Climbing_Turn",
	"Descending_Turn",
	"Dive",
	"Zoom_Climb",
	"Large_Heading_Change",
	"Oscillatory_Maneuver",
	"Transition",
	"Uncertain",
}

local LABEL_IDS = {}
for labelId, labelName in ipairs(MAIN_LABELS) do
	LABEL_IDS[labelName] = labelId - 1
end

local propertyIndexCache = {}
local menuIds = {}

local function JoinPath(basePath, childName)
	if basePath == nil or basePath == "" then
		return childName
	end

	local separator = string.sub(basePath, -1)
	if separator == "/" or separator == "\\" then
		return basePath .. childName
	end

	return basePath .. "/" .. childName
end

local addonPath = Tacview.AddOns.Current.GetPath() or "."
local configPath = JoinPath(addonPath, "manual_label_editor_path.txt")
local exportCsvPath = JoinPath(addonPath, "manual_label_edits.csv")

local function Trim(value)
	if value == nil then
		return ""
	end

	return tostring(value):match("^%s*(.-)%s*$")
end

local function EscapeCsv(value)
	local text = tostring(value or "")
	text = string.gsub(text, "\"", "\"\"")
	return "\"" .. text .. "\""
end

local function ReadTextFile(path)
	local file = io.open(path, "r")
	if file == nil then
		return nil
	end

	local content = file:read("*a")
	file:close()
	return content
end

local function WriteTextFile(path, content)
	local file = io.open(path, "w")
	if file == nil then
		return false
	end

	file:write(content)
	file:close()
	return true
end

local function LoadExportPath()
	local content = ReadTextFile(configPath)
	if content == nil then
		return
	end

	local configuredPath = Trim(content)
	if configuredPath ~= "" then
		exportCsvPath = configuredPath
	end
end

local function SaveExportPath(path)
	exportCsvPath = path
	WriteTextFile(configPath, path)
end

local function EnsureParentDirectoryExists(path)
	local directory = string.match(path, "^(.*)[/\\][^/\\]+$")
	if directory == nil or directory == "" then
		return
	end

	os.execute('mkdir "' .. directory .. '" >nul 2>nul')
end

local function EnsureCsvHeader(path)
	local existing = ReadTextFile(path)
	if existing ~= nil and existing ~= "" then
		return
	end

	EnsureParentDirectoryExists(path)
	local file = io.open(path, "w")
	if file == nil then
		Tacview.Log.Error("Unable to open manual label export file: ", path)
		return
	end

	file:write("edit_index,edit_action,edited_utc,aircraft_id,aircraft_name,sample_time,window_start_hint,window_end_hint,manual_label_name,manual_label_id,previous_predicted_label,previous_rule_label,previous_main_label,addon_version\n")
	file:close()
end

local function AppendCsvRow(row)
	EnsureCsvHeader(exportCsvPath)
	local file = io.open(exportCsvPath, "a")
	if file == nil then
		Tacview.Log.Error("Unable to append manual label export file: ", exportCsvPath)
		return
	end

	local cells =
	{
		EscapeCsv(row.edit_index),
		EscapeCsv(row.edit_action),
		EscapeCsv(row.edited_utc),
		EscapeCsv(row.aircraft_id),
		EscapeCsv(row.aircraft_name),
		EscapeCsv(row.sample_time),
		EscapeCsv(row.window_start_hint),
		EscapeCsv(row.window_end_hint),
		EscapeCsv(row.manual_label_name),
		EscapeCsv(row.manual_label_id),
		EscapeCsv(row.previous_predicted_label),
		EscapeCsv(row.previous_rule_label),
		EscapeCsv(row.previous_main_label),
		EscapeCsv(row.addon_version),
	}

	file:write(table.concat(cells, ","))
	file:write("\n")
	file:close()
end

local function GetTextPropertyIndex(name, autoCreate)
	local index = propertyIndexCache[name]
	if index ~= nil then
		return index
	end

	index = Tacview.Telemetry.GetObjectsTextPropertyIndex(name, autoCreate)
	propertyIndexCache[name] = index
	return index
end

local function GetNumericPropertyIndex(name, autoCreate)
	local index = propertyIndexCache[name]
	if index ~= nil then
		return index
	end

	index = Tacview.Telemetry.GetObjectsNumericPropertyIndex(name, autoCreate)
	propertyIndexCache[name] = index
	return index
end

local function GetTextSample(objectHandle, absoluteTime, propertyName)
	local propertyIndex = GetTextPropertyIndex(propertyName, false)
	if propertyIndex == nil or propertyIndex == Tacview.Telemetry.InvalidPropertyIndex then
		return ""
	end

	local value = Tacview.Telemetry.GetTextSample(objectHandle, absoluteTime, propertyIndex)
	if value == nil then
		return ""
	end

	return Trim(value)
end

local function GetNumericSample(objectHandle, absoluteTime, propertyName)
	local propertyIndex = GetNumericPropertyIndex(propertyName, false)
	if propertyIndex == nil or propertyIndex == Tacview.Telemetry.InvalidPropertyIndex then
		return nil
	end

	local value = Tacview.Telemetry.GetNumericSample(objectHandle, absoluteTime, propertyIndex)
	return value
end

local function SetTextSample(objectHandle, absoluteTime, propertyName, value)
	local propertyIndex = GetTextPropertyIndex(propertyName, true)
	if propertyIndex == nil or propertyIndex == Tacview.Telemetry.InvalidPropertyIndex then
		return
	end

	Tacview.Telemetry.SetTextSample(objectHandle, absoluteTime, propertyIndex, value)
end

local function SetNumericSample(objectHandle, absoluteTime, propertyName, value)
	local propertyIndex = GetNumericPropertyIndex(propertyName, true)
	if propertyIndex == nil or propertyIndex == Tacview.Telemetry.InvalidPropertyIndex then
		return
	end

	Tacview.Telemetry.SetNumericSample(objectHandle, absoluteTime, propertyIndex, value)
end

local function GetCurrentAbsoluteTime(objectHandle)
	local transform = Tacview.Telemetry.GetCurrentTransform(objectHandle)
	if transform ~= nil and transform.time ~= nil then
		return transform.time
	end

	local firstTransformTime, lastTransformTime = Tacview.Telemetry.GetTransformTimeRange(objectHandle)
	if lastTransformTime ~= nil then
		return lastTransformTime
	end
	if firstTransformTime ~= nil then
		return firstTransformTime
	end

	local dataStartTime, _ = Tacview.Telemetry.GetDataTimeRange()
	if dataStartTime ~= nil then
		return dataStartTime
	end

	return 0
end

local function IsFixedWingObject(objectHandle)
	local tags = Tacview.Telemetry.GetCurrentTags(objectHandle)
	if tags == nil then
		return false
	end

	return Tacview.Telemetry.AllGivenTagsActive(
		tags,
		Tacview.Telemetry.Tags.Air + Tacview.Telemetry.Tags.FixedWing
	)
end

local function FormatUtcTimestamp()
	return os.date("!%Y-%m-%dT%H:%M:%SZ")
end

local function NextEditIndex()
	local existing = ReadTextFile(exportCsvPath)
	if existing == nil or existing == "" then
		return 0
	end

	local count = 0
	for _ in string.gmatch(existing, "\n") do
		count = count + 1
	end

	return math.max(0, count - 1)
end

local function ApplyManualLabel(objectHandle, manualLabelName, editAction)
	if objectHandle == nil or objectHandle == 0 then
		return
	end

	local absoluteTime = GetCurrentAbsoluteTime(objectHandle)
	local objectId = Tacview.Telemetry.GetObjectId(objectHandle) or ""
	local aircraftName = Tacview.Telemetry.GetCurrentShortName(objectHandle) or ""
	local previousPredictedLabel = GetTextSample(objectHandle, absoluteTime, "Maneuver010PredLabel")
	local previousRuleLabel = GetTextSample(objectHandle, absoluteTime, "Maneuver020RuleLabel")
	local previousMainLabel = GetTextSample(objectHandle, absoluteTime, "Maneuver030MainLabel")
	local sampleTimeHint = GetNumericSample(objectHandle, absoluteTime, "Maneuver072SampleTime")
	local windowStartHint = GetNumericSample(objectHandle, absoluteTime, "Maneuver070WindowStart")
	local windowEndHint = GetNumericSample(objectHandle, absoluteTime, "Maneuver071WindowEnd")

	SetTextSample(objectHandle, absoluteTime, "Maneuver015HumanLabel", manualLabelName or "")
	SetNumericSample(objectHandle, absoluteTime, "Maneuver016HumanLabelId", LABEL_IDS[manualLabelName] or -1)
	SetTextSample(objectHandle, absoluteTime, "Maneuver017HumanAction", editAction)
	SetTextSample(objectHandle, absoluteTime, "Maneuver018HumanEditedUtc", FormatUtcTimestamp())
	SetTextSample(objectHandle, absoluteTime, "Maneuver019HumanSource", "TacviewAddon")

	AppendCsvRow(
		{
			edit_index = NextEditIndex(),
			edit_action = editAction,
			edited_utc = FormatUtcTimestamp(),
			aircraft_id = objectId,
			aircraft_name = aircraftName,
			sample_time = sampleTimeHint or absoluteTime,
			window_start_hint = windowStartHint or "",
			window_end_hint = windowEndHint or "",
			manual_label_name = manualLabelName or "",
			manual_label_id = LABEL_IDS[manualLabelName] or "",
			previous_predicted_label = previousPredictedLabel,
			previous_rule_label = previousRuleLabel,
			previous_main_label = previousMainLabel,
			addon_version = ADDON_VERSION,
		}
	)

	Tacview.Log.Info(
		"[ManualLabelEditor] ",
		editAction,
		" ",
		manualLabelName or "<cleared>",
		" for object ",
		tostring(objectId),
		" at time ",
		tostring(sampleTimeHint or absoluteTime),
		" -> ",
		exportCsvPath
	)
end

local function ChooseExportCsv()
	local selectedPath = Tacview.UI.MessageBox.GetSaveFileName(
		{
			defaultFileExtension = "csv",
			fileName = exportCsvPath,
			fileTypeList =
			{
				{ ".csv", "Comma-separated values" }
			}
		}
	)

	if selectedPath == nil then
		return
	end

	SaveExportPath(selectedPath)
	Tacview.Log.Info("[ManualLabelEditor] Manual label CSV path set to ", exportCsvPath)
end

local function ShowCurrentExportPath()
	Tacview.Log.Info("[ManualLabelEditor] Manual label CSV path: ", exportCsvPath)
end

local function BuildTopMenu()
	menuIds.root = Tacview.UI.Menus.AddMenu(nil, "Manual Label Editor")
	Tacview.UI.Menus.AddCommand(menuIds.root, "Choose edits CSV...", ChooseExportCsv)
	Tacview.UI.Menus.AddCommand(menuIds.root, "Log current edits CSV path", ShowCurrentExportPath)
end

local function BuildContextMenu(contextMenuId, objectHandle)
	if objectHandle == nil or objectHandle == 0 then
		return
	end

	if not IsFixedWingObject(objectHandle) then
		return
	end

	local parentMenuId = Tacview.UI.Menus.AddMenu(contextMenuId, "Manual Label")

	for _, labelName in ipairs(MAIN_LABELS) do
		Tacview.UI.Menus.AddCommand(
			parentMenuId,
			labelName,
			function()
				ApplyManualLabel(objectHandle, labelName, "set")
			end
		)
	end

	Tacview.UI.Menus.AddSeparator(parentMenuId)
	Tacview.UI.Menus.AddCommand(
		parentMenuId,
		"Clear Human Label",
		function()
			ApplyManualLabel(objectHandle, "", "clear")
		end
	)
end

LoadExportPath()
BuildTopMenu()
Tacview.UI.Renderer.ContextMenu.RegisterListener(BuildContextMenu)
Tacview.Log.Info("[ManualLabelEditor] Ready. Export CSV: ", exportCsvPath)
