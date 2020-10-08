addpath(genpath('PQ_tools'));

%% define socket
sock = define_socket;

%% start empty world
msg_received = MC('START', 5, sock);

%% start hunting and gathering
msg_received = MC('RESET domain ../experiments/hgv1_1.json', 5, sock);



%%
MC('SMOOTH_TILT FORWARD', 0.1, sock); % initialize pitch 
MC('LOOK_EAST', 0.1, sock); % initialize pitch 



%% get all info from the game "SENSE_ALL NONAV"
% this is more for debugging purposes, the bot shouldn't use this API to get information from the game
    msg_received = MC('SENSE_ALL NONAV', 0.3, sock);
    s = jsondecode(msg_received);
    block_coordinates = []; 
    block_content = cell(0);
    for field_name = fieldnames(s.map)'
        block_name = field_name{1};
        block_coordinates = [block_coordinates,str2double(regexp(block_name(2:end),'_','split'))'];
        tmp = getfield(s.map,block_name);
        block_content = [block_content, {tmp.name}];
    end
    macguffin_coordinates = block_coordinates(:,find(ismember(block_content,'polycraft:macguffin')));
    msg_received = MC('SENSE_LOCATIONS', 0.1, sock); s = jsondecode(msg_received);
    destination_coordinates = s.destinationPos;
    player_coordinates = s.player.pos;
    player_yaw = s.player.yaw;
    tmp = rotation_matrix_2D(s.player.yaw)*[0;1]; player_directions = [tmp(1,1);0;tmp(2,1)];
    player_coordinates_history = [s.player.pos];


    game_state_structure = struct('block_coordinates', [], ...
                                  'block_content', [], ...
                                  'macguffin_coordinates', [], ...
                                  'destination_coordinates',[],...
                                  'player_coordinates',[], ...
                                  'player_yaw', [], ...
                                  'player_directions', [], ...
                                  'player_coordinates_history', []);

    game_state_structure.block_coordinates = block_coordinates;
    game_state_structure.block_content = block_content;
    game_state_structure.macguffin_coordinates = macguffin_coordinates;
    game_state_structure.destination_coordinates = destination_coordinates;
    game_state_structure.player_coordinates = player_coordinates;
    game_state_structure.player_yaw = player_yaw;
    game_state_structure.player_directions = player_directions;
    game_state_structure.player_coordinates_history = player_coordinates_history;

    display_HG_map(game_state_structure)

    
%% allowed APIs for the AI-bot to get information from game

    % sense player location, but not use the destination coordinates
    msg_received = MC('SENSE_LOCATIONS', 0.1, sock); s = jsondecode(msg_received);
    player_coordinates = s.player.pos;
    player_yaw = s.player.yaw;
    tmp = rotation_matrix_2D(s.player.yaw)*[0;1]; player_directions = [tmp(1,1);0;tmp(2,1)];

    % sense screen, which is similar to capture screenshot of the game
    msg_received = MC('SENSE_SCREEN', 0.3, sock); s = jsondecode(msg_received);
    
    tmp = reshape(typecast(uint32(s.screen.img+2^32),'uint8'),[4,65536])';
    screenshot = reshape(fliplr(tmp(:,1:3)),[256,256,3]);
    screenshot = cat(3, flipud(reshape(tmp(:,3),[256,256])'), flipud(reshape(tmp(:,2),[256,256])'), flipud(reshape(tmp(:,1),[256,256])'));
    


