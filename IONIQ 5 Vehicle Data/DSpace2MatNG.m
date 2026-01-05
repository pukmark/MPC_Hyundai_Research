function Data = DSpace2MatNG( filename )

%set modify to default value when not specified
if ismac
    if (nargin < 1)
        [FileName,PathName] = uigetfile('data/*.mat', 'Select dspace file');
        filename = [PathName FileName];
    end
    %only path given
    if (isempty(strfind(filename(end-2:end),'mat')))
        [FileName,PathName] = uigetfile([filename '/*.mat'], 'Select dspace file');
        filename = [PathName FileName];
    end
else
    if (nargin < 1)
        [FileName,PathName] = uigetfile('*.mat', 'Select dspace file');
        filename = [FileName];
    end
    %only path given
    if (isempty(strfind(filename(end-2:end),'mat')))
        [FileName,PathName] = uigetfile([filename '/*.mat'], 'Select dspace file');
        filename = [FileName];
    end
end

%load file
DSpace=load(filename);  tmp = fieldnames(DSpace); DSpace = DSpace.(tmp{1});  
Data = [];
Data.FileName = filename;

%length of dspace channels
if length(DSpace.X) > 1
    Data.Time = DSpace.X(end-1).Data; % end is replace with end-1 due to some part is on change
else
    Data.Time = DSpace.X(end).Data;
end
%number of dspace channels
num = length(DSpace.Y);
originalLength = num;

%loop through channels
for i=1:num
    channel.name = DSpace.Y(i).Name;
    if  (strcmp(DSpace.Y(i).Name,'Value') == 1)||(strcmp(DSpace.Y(i).Name,'Out1') == 1)||(strcmp(DSpace.Y(i).Name,'y') == 1)||(strcmp(DSpace.Y(i).Name,'out_torque') == 1)
            DSpace.Y(i).Path(12:end);
            channel.name = DSpace.Y(i).Path(12:end);
            
            if strcmp('1-D Lookup\nTable2',channel.name), channel.name = 'v_des'; end
            if strcmp('CalculatingTmin',channel.name), channel.name = 'TmaftCrp'; end
            if strcmp('ClockReset1',channel.name), channel.name = 'time_spent'; end
            if strcmp('ClockReset2',channel.name), channel.name = 'trav_dist'; end
            if strcmp('Saturation1',channel.name), channel.name = 'Brk_Cmd'; end
            if strcmp('Saturation2',channel.name), channel.name = 'Tm_Cmd'; end
            if strcmp('Saturation3',channel.name), channel.name = 'Slidin_Cmd_Sat'; end
            if strcmp('km//hr to m//s',channel.name), channel.name = 'VehSpd'; end
            if strcmp('SubtractingCrpTq',channel.name), channel.name = 'Tm_goinginPT'; end
            if strcmp('Saturation\nDynamic',channel.name), channel.name = 'Tm_Cmd_Trimmed'; end
            
           channel.name= channel.name(find(~isspace(channel.name)));
    end
    try
    Data.(channel.name) =DSpace.Y(i).Data;
    end
end