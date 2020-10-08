% connect to socket'
HOST = '127.0.0.1';
PORT = 9000;
sock = tcpip(HOST, PORT);

msg_received = MC('START', 5, sock)
% msg_received = MC('RESET domain ../experiments/pogo.psm', 5, sock)
msg_received = MC('RESET domain ../experiments/hgv1_1.json', 5, sock)
% s = jsondecode(msg_received)
% s.recipes.inputs
% s.recipes.outputs


%%
% msg_received = MC('SMOOTH_MOVE W', 0.1, sock)
% s = jsondecode(msg_received)
% s.goal
% s.command_result


%%
% msg_received = MC('SENSE_ALL', 0.1, sock)
% s = jsondecode(msg_received)


%%
msg_received = MC('SENSE_ALL NONAV', 0.1, sock)
s = jsondecode(msg_received)

%%
% MC('TP_TO 5,6,5')
% MC('MOVE_WEST')
% MC('MOVE_NORTH')
% MC('MOVE_EAST')
% MC('MOVE_SOUTH')
% MC('SMOOTH_MOVE W')
% MC('SMOOTH_MOVE A')
% MC('SMOOTH_MOVE D')
% MC('SMOOTH_MOVE X')
% MC('SMOOTH_MOVE Q')
% MC('SMOOTH_MOVE E')
% MC('SMOOTH_MOVE Z')
% MC('SMOOTH_MOVE C')
% MC('SMOOTH_TURN 90')
% MC('SMOOTH_T9iolILT DOWN')    # FORWARD
%  
% MC('BREAK_BLOCK')
% MC('PLACE_CRAFTING_TABLE')
% MC('PLACE_TREE_TAP')
% MC('EXTRACT_RUBBER')
% MC('CRAFT 1 minecraft:log 0 0 0') # needs a log | create a plank
% MC('CRAFT 1 minecraft:planks 0 minecraft:planks 0') # needs 2 planks | create 4 sticks
% MC('CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0') # needs 4 planks and 1 stick | create 1 tree tap
% MC('CRAFT 1 minecraft:planks minecraft:stick minecraft:planks minecraft:planks 0 minecraft:planks 0 minecraft:planks 0') # needs 4 planks and 1 stick | create 1 tree tap
% MC('CRAFT 1 minecraft:stick minecraft:stick minecraft:stick minecraft:planks minecraft:stick minecraft:planks 0 polycraft:sack_polyisoprene_pellets 0') # needs 2 planks, 4 sticks and 1 rubber sac| create 1 pogo stick
%  
% MC('SENSE_INVENTORY')
% MC('SENSE_RECIPE')
% MC('SENSE_ALL')


%%
function msg_received = MC(msg_send, time_delay, sock)

    fopen(sock);
    fwrite(sock,unicode2native([msg_send,char(10)]))
    pause(time_delay)

    msg_received = [];
    while sock.BytesAvailable~=0
        current_BytesAvailable = sock.BytesAvailable;
        bytes = fread(sock, [1, sock.BytesAvailable]);
        msg_received = [msg_received, char(bytes)];
        pause(0.1)
    end
    
    fclose(sock)
end