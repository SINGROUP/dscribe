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
local file = io.open(arg[1], "r")
-- local file = io.open("phi_raw.tex", "r")
file:seek("set",0)
-- clearning unnessesary string
local longText = file:read("*a")
local longText = longText:gsub("&", "\n")
local longText = longText:gsub("...begin.pmatrix.", "")
local longText = longText:gsub(".end.pmatrix..", "")
local longText = longText:gsub("{x}", "x")
local longText = longText:gsub("{y}", "y")
local longText = longText:gsub("{z}", "z")
local longText = longText:gsub("%s%=%s", "%=")
print(longText)

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
local phi = "\\Phi"
local phi = "\\frac{\\partial Phi}{\\partial x}"
local phi = "\\frac{\\partial Phi}{\\partial y}"
local phi = "\\frac{\\partial Phi}{\\partial z}"
for i, v in ipairs(t) do
    if m <= l then
	    if string.find(string.lower(arg[1]),"dx")  then
		    table.insert(tfunc, "\\frac{ \\partial \\Phi_{" .. l .. " , " .. m .. "}}{\\partial x} =" .. v)
	    elseif string.find(string.lower(arg[1]),"dy")  then
		    table.insert(tfunc, "\\frac{ \\partial \\Phi_{" .. l .. " , " .. m .. "}}{\\partial y} =" .. v)
	    elseif string.find(string.lower(arg[1]),"dz")  then
		    table.insert(tfunc, "\\frac{ \\partial \\Phi_{" .. l .. " , " .. m .. "}}{\\partial z} =" .. v)
	    else
		    table.insert(tfunc, "\\Phi" .. "_{" .. l .. " , " .. m .. "} =" .. v)
    end
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
