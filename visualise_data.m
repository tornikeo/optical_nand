clear;
close all;
clc;


% load('full_rec_2__at_radius_28.mat', ...
%     'middle_guide_recording',...
%     'upper_guide_recording',...
%     'NCELL'...
%     );

% load('full_rec_2_both.mat', ...
%     'middle_guide_recording',...
%     'upper_guide_recording',...
%     'NCELL');

load('full_rec_3_xl_230_brd.mat', ...
    'middle_guide_recording',...
    'upper_guide_recording',...
    'cases_brd_dim',...
    'bridge_cylinder_diamters',...
    'NCELL');

cyl_diam = bridge_cylinder_diamters(1) / NCELL(1)

plot(upper_guide_recording(809 + 0 * NCELL(1),:))

% for t = 1:size(data_1,2)-1
%     plot(data_1_ax,data_1(data_1_ax,t), '-b')
%     hold on
%     plot(data_2_ax,data_2(data_2_ax,t), '-g')
%     hold off
%     title(['T: ' int2str(t)]);
%     axis tight
%     ylim([-300, 300])
%     drawnow()
% end
% plot(middle_guide_recording(521+29.0*NCELL(1), :))
% plot(upper_guide_recording(end/4, :))


% ERx(ports.p1.x, ports.p1.y) = 15;
% ERy(ports.p2.x, ports.p2.y) = 15;
% 
% ERx(Nx_src_lo_AND:Nx_src_hi_AND, ...
%     Ny_src_lo_AND-1:Ny_src_lo_AND+1) = -10;
% ERy(Nx_src_lo_NOT:Nx_src_hi_NOT, ...
%     Ny_src_lo_NOT-1:Ny_src_lo_NOT+1) = -10;
% 
% ERx(rec_line_AND.x, rec_line_AND.y) = 0;
% ERx(rec_line_NOT.x, rec_line_NOT.y) = 0;
% 
% imagesc(log(1+sigx(1:2:Nx2,2:2:Ny2)+sigy(1:2:Nx2,2:2:Ny2)) + ERx + ERy + log(1e-4+PECx + PECy) +...
%         + 0.5.*(ERy+ERx+2.*PECx+2.*PECy));
% axis image;
% return;