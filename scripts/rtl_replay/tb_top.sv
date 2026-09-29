// RTL replay of one bfpp_acc v3 MUL_MAT tile through the packaged IP's top
// (bfp_acc_top: AXI-Stream adapters and FIFOs), as the DMAs drive it on the
// board: TLAST only on the last word of each transfer. SECDA-LLM TODO 3.7.
`timescale 1ns/1ps
module tb;
  reg aclk = 0, aresetn = 0;
  always #2.5 aclk = ~aclk; // 200 MHz
  integer cycle = 0;
  always @(posedge aclk) cycle <= cycle + 1;

  reg [31:0] m0 [0:4095]; reg [31:0] m1 [0:4095]; reg [31:0] m2 [0:4095]; reg [31:0] m3 [0:4095];
  integer head[0:3], tail[0:3];
  wire [3:0] tv = {head[3]<tail[3], head[2]<tail[2], head[1]<tail[1], head[0]<tail[0]};
  wire [3:0] tl = {head[3]==tail[3]-1, head[2]==tail[2]-1, head[1]==tail[1]-1, head[0]==tail[0]-1};
  wire [31:0] td0 = m0[tv[0] ? head[0] : 0], td1 = m1[tv[1] ? head[1] : 0],
              td2 = m2[tv[2] ? head[2] : 0], td3 = m3[tv[3] ? head[3] : 0];
  wire [3:0] tr;
  reg out_armed = 0;
  wire o_valid; wire [31:0] o_data; wire [0:0] o_last;

  bfp_acc_top dut (
    .M_AXIS_DATA1_TVALID(o_valid), .M_AXIS_DATA1_TREADY(out_armed), .M_AXIS_DATA1_TDATA(o_data), .M_AXIS_DATA1_TLAST(o_last),
    .M_AXIS_DATA2_TVALID(), .M_AXIS_DATA2_TREADY(1'b1), .M_AXIS_DATA2_TDATA(), .M_AXIS_DATA2_TLAST(),
    .M_AXIS_DATA3_TVALID(), .M_AXIS_DATA3_TREADY(1'b1), .M_AXIS_DATA3_TDATA(), .M_AXIS_DATA3_TLAST(),
    .M_AXIS_DATA4_TVALID(), .M_AXIS_DATA4_TREADY(1'b1), .M_AXIS_DATA4_TDATA(), .M_AXIS_DATA4_TLAST(),
    .S_AXIS_DATA1_TVALID(tv[0]), .S_AXIS_DATA1_TREADY(tr[0]), .S_AXIS_DATA1_TDATA(td0), .S_AXIS_DATA1_TLAST(tl[0]),
    .S_AXIS_DATA2_TVALID(tv[1]), .S_AXIS_DATA2_TREADY(tr[1]), .S_AXIS_DATA2_TDATA(td1), .S_AXIS_DATA2_TLAST(tl[1]),
    .S_AXIS_DATA3_TVALID(tv[2]), .S_AXIS_DATA3_TREADY(tr[2]), .S_AXIS_DATA3_TDATA(td2), .S_AXIS_DATA3_TLAST(tl[2]),
    .S_AXIS_DATA4_TVALID(tv[3]), .S_AXIS_DATA4_TREADY(tr[3]), .S_AXIS_DATA4_TDATA(td3), .S_AXIS_DATA4_TLAST(tl[3]),
    .s_axi_ctrl_AWADDR(7'd0), .s_axi_ctrl_AWVALID(1'b0), .s_axi_ctrl_AWREADY(), .s_axi_ctrl_WDATA(32'd0), .s_axi_ctrl_WSTRB(4'd0),
    .s_axi_ctrl_WVALID(1'b0), .s_axi_ctrl_WREADY(), .s_axi_ctrl_BRESP(), .s_axi_ctrl_BVALID(), .s_axi_ctrl_BREADY(1'b1),
    .s_axi_ctrl_ARADDR(7'd0), .s_axi_ctrl_ARVALID(1'b0), .s_axi_ctrl_ARREADY(), .s_axi_ctrl_RDATA(), .s_axi_ctrl_RRESP(),
    .s_axi_ctrl_RVALID(), .s_axi_ctrl_RREADY(1'b1),
    .s_axi_hwc_AWADDR(9'd0), .s_axi_hwc_AWVALID(1'b0), .s_axi_hwc_AWREADY(), .s_axi_hwc_WDATA(32'd0), .s_axi_hwc_WSTRB(4'd0),
    .s_axi_hwc_WVALID(1'b0), .s_axi_hwc_WREADY(), .s_axi_hwc_BRESP(), .s_axi_hwc_BVALID(), .s_axi_hwc_BREADY(1'b1),
    .s_axi_hwc_ARADDR(9'd0), .s_axi_hwc_ARVALID(1'b0), .s_axi_hwc_ARREADY(), .s_axi_hwc_RDATA(), .s_axi_hwc_RRESP(),
    .s_axi_hwc_RVALID(), .s_axi_hwc_RREADY(1'b1),
    .aresetn(aresetn), .aclk(aclk)
  );

  integer c;
  always @(posedge aclk)
    for (c = 0; c < 4; c = c + 1) if (tv[c] && tr[c]) head[c] <= head[c] + 1;

  integer nout = 0, nbad = 0, fd, expw, expl, t_op = 0, t_first = 0, t_last = 0, i, k;
  reg [31:0] exp_data [0:1023];
  integer gaps [0:1023];
  always @(posedge aclk) if (o_valid && out_armed) begin
    if (nout == 0) t_first = cycle;
    if (nout > 0 && nout < 1024) gaps[nout] = cycle - t_last;
    t_last = cycle;
    if (o_data !== exp_data[nout]) begin
      if (nbad < 5) $display("MISMATCH out %0d: rtl %08x sim %08x", nout, o_data, exp_data[nout]);
      nbad = nbad + 1;
    end
    nout = nout + 1;
  end

  task load(input integer p, input integer chn, input integer n);
    reg [8*64-1:0] fn;
    begin
      $sformat(fn, "p%0d_c%0d.hex", p, chn);
      case (chn)
        0: $readmemh(fn, m0, 0, n - 1);
        1: $readmemh(fn, m1, 0, n - 1);
        2: $readmemh(fn, m2, 0, n - 1);
        3: $readmemh(fn, m3, 0, n - 1);
      endcase
      head[chn] = 0; tail[chn] = n;
    end
  endtask
  function integer drained(input integer dummy);
    drained = (head[0] >= tail[0]) && (head[1] >= tail[1]) && (head[2] >= tail[2]) && (head[3] >= tail[3]);
  endfunction

  initial begin
    for (i = 0; i < 4; i = i + 1) begin head[i] = 0; tail[i] = 0; end
    fd = $fopen("expected.hex", "r");
    for (i = 0; i < 512; i = i + 1) begin k = $fscanf(fd, "%h %d", expw, expl); exp_data[i] = expw; end
    $fclose(fd);
    repeat (16) @(posedge aclk); aresetn = 1;
    repeat (20) @(posedge aclk);
    load(0, 0, 298); @(posedge aclk); while (!drained(0)) @(posedge aclk); @(posedge aclk);
    $display("PHASE input packet done at cycle %0d", cycle);
    load(1, 0, 7); @(posedge aclk); while (!drained(0)) @(posedge aclk); @(posedge aclk);
    $display("PHASE weight opcode done at cycle %0d", cycle);
    load(2, 0, 672); load(2, 1, 672); load(2, 2, 672); load(2, 3, 672); @(posedge aclk); while (!drained(0)) @(posedge aclk); @(posedge aclk);
    $display("PHASE weights done at cycle %0d", cycle);
    t_op = cycle;
    load(3, 0, 5); @(posedge aclk); while (!drained(0)) @(posedge aclk); @(posedge aclk);
    $display("PHASE compute opcode done at cycle %0d", cycle);
    out_armed = 1;
    fork
      wait (nout == 512);
      begin repeat (100000) @(posedge aclk); $display("TIMEOUT nout=%0d", nout); end
    join_any
    repeat (5) @(posedge aclk);
    $display("RESULT outputs=%0d mismatches=%0d", nout, nbad);
    $display("RESULT compute_op_cycle=%0d first_out=%0d last_out=%0d span=%0d per_output=%0.2f",
             t_op, t_first, t_last, t_last - t_op, (t_last - t_op) / 512.0);
    for (i = 1; i < 12; i = i + 1) $write("%0d ", gaps[i]); $display(" <- cycles between outputs 1..11");
    for (i = 120; i < 136; i = i + 1) $write("%0d ", gaps[i]); $display(" <- outputs 120..135");
    $finish;
  end
endmodule
