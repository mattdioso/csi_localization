load data.mat -ascii
BW = 80;

clear alldata;

FREQ = 2^26 / 277022;
FREQ = FREQ * 1e6;

% extract all SN
SN = unique (sort (data (:, 1)));

% guess cores and nsss and items per spectrum
cores = unique (sort (data (:, 2)));
nsss = unique (sort (data (:, 3)));
tonesmin = unique (sort (data (:, 4)));

% we must check that for each captured csi we have the same type
% of data, e.g., same number of cores/nss/bandwidths
coreno = length (cores);
nssno = length (nsss);
itemsn = coreno * nssno * 1;
itemperspectrum = length (tonesmin);
itemsn = itemsn * itemperspectrum;

txtsdata = [];

% process all SN separately and keep data only if we have for all core/nss/spectral width
packet = 1;
for snjj = 1:length (SN),
  clear framecsi;

  sn = SN (snjj);
  datasn = data (find (data (:, 1) == sn), :);
  if size (datasn, 1) ~= itemsn,
    disp (sprintf ('Missing data for SN %d', sn));
  end;

  error = 0;
  ts = min (datasn (:, 13));
  rxts = min (datasn (:, 14));
  rxtsslow = min (datasn (:, 15));
  rxseqno = min (datasn (:, 16));
  txts = min (datasn (:, 17));
  txseqno = min (datasn (:, 18));
  powers = datasn (:, 19:22);
  if sum (sum (diff (powers, [], 1))) ~= 0,
    disp 'Powers not constant over profiles of the same frame';
    keyboard
  end;
  phy0 = datasn (:, 23);
  if sum (diff (phy0, [], 1)) ~= 0,
    disp 'phystats not constant over profiles of the same frame';
  end;

  txtsdata = [txtsdata; [txts txseqno]];

  % process data, give error if missing
  for corejj = 1:length (cores),
    core = cores (corejj);
    for nsskk = 1:length (nsss),
      nss = nsss (nsskk);
  
      datasub = datasn (find (datasn (:, 2) == core & datasn (:, 3) == nss), :);

      if size (datasub, 1) ~= itemperspectrum,
        disp (sprintf ('incomplete data for sn %d', sn));
        error = 1;
        break;
      end;

      tonemin = min (datasub (:, 4));
      tonemax = max (datasub (:, 5));

      phytype = unique (datasub (:, 6));
      if length (phytype) > 1,
        disp (sprintf ('multiple phytype for sn %d', sn));
        error = 1;
        break;
      end;

      macaddr = datasub (7:12);
      datasub = datasub (:, 28:2:end) + j * datasub (:, 29:2:end);
      N = size (datasub, 2);

      if itemperspectrum > 1,
        disp (sprintf ('itemperspectrum > 1 unsupported'));
        error = 1;
        break;

        els = size (datasub, 1) * size (datasub, 2);
        datasub = reshape (transpose (datasub), 1, els);
      end;

      framecsi.core{core + 1}.nss{nss + 1}.data = datasub;

    end;
    if error == 1,
      break;
    end;
  end;

  framecsi.tones = [tonemin tonemax];
  framecsi.phytype = phytype;
  framecsi.rxts = int64 (rxts);
  framecsi.rxtsslow = rxtsslow;
  framecsi.rxseqno = rxseqno;
  framecsi.txts = int64 (0);
  framecsi.sn = sn;
  framecsi.ts = ts;
  framecsi.srcmac = sprintf ('%02X%02X%02X%02X%02X%02X', macaddr);
  framecsi.powers = powers (1, :);
  framecsi.phy0 = phy0 (1, :);
  

  alldata (packet) = framecsi;

  % snjj,
  packet = packet + 1;
end;

packets = packet - 1;

disp (sprintf ('Found CSI for %d packets\n', packets));
