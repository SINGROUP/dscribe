-- -- file manipulation
-- -- r: read only (default)
-- -- w: Overwrite or create a new file
-- -- a: append or create a new file
-- -- r+: read & write existing file
-- -- w+: Overwrite read or create a file
-- -- a+: Append read or create file
-- file = io.open("test.lua", "w+")
-- file:write("random string of text \n")
-- file:write(" more random string of text \n")
--
-- file:seek("set", 0)
-- print(file:read("*a"))
-- file:close()
--
-- file = io.open("test.tex", "a+")
-- file:write("EVEN MOIRE TEXT\n")
-- file:seek("set",0)
-- print(file:read("*a"))
-- file:close()
--
local file = io.open("test.tex", "r")
-- local file = io.open("phi_raw.tex", "r")
file:seek("set",0)
-- clearning unnessesary string
local longText = file:read("*a")
local longText = longText:gsub("&", "\n")
local longText = longText:gsub("^.*pmatrix.", "")
local longText = longText:gsub("{x}", "x")
local longText = longText:gsub("{y}", "y")
local longText = longText:gsub("{z}", "z")
local longText = longText:gsub("%s%=%s", "%=")

local t = {}
local j = 0
for line in string.gmatch(longText, string.format("([^\n]*)\n")) do
  table.insert(t, line)
  -- print(line)
end

-- extracting only the relevant components
local tfunc = {}
local l = 0
local m = 0
local reset = 0
for i, v in ipairs(t) do
    if m <= l then
	    table.insert(tfunc, "\\Phi_{" .. l .. " , " .. m .. "} =" .. v)
    end
    m = m + 1
    reset = reset + 1
    if reset == 41 then -- 40 columns
	    l = l + 1
	    m = -l
	    reset = 0
    end
end

--printing results
for i, v in ipairs(tfunc) do
  print(v)
end


-- print(longText)
file:close()
