function [ vect ] = linspRound( loc, step, dist )
    vect = round( ((dist.*loc-dist.*step):(dist.*loc+dist.*step)));
end

