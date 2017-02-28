import datetime
import numpy as np
import matplotlib.pyplot as plt
try:
     from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
     # quotes_historical_yahoo_ochl was named quotes_historical_yahoo before matplotlib 1.4
     from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ochl
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

###############################################################################
# Retrieve the data from Internet

# Choose a time period reasonably calm (not too long ago so that we get
# high-tech firms, and before the 2008 crash)
d1 = datetime.datetime(2016, 1, 1)
d2 = datetime.datetime(2017, 1, 1)

# kraft symbol has now changed from KFT to MDLZ in yahoo
symbol_dict = {
    'CBA.AX': 'Commonwealth Bank of Australia', 
    'WBC.AX': 'Westpac Banking Corp.', 
    'ANZ.AX': 'Australia & New Zealand Banking Group Ltd.', 
    'BHP.AX': 'BHP Billiton Ltd.', 
    'NAB.AX': 'National Australia Bank Ltd.', 
    'TLS.AX': 'Telstra Corp. Ltd.', 
    'WES.AX': 'Wesfarmers Ltd.', 
    'CSL.AX': 'CSL Ltd.', 
    'WOW.AX': 'Woolworths Ltd.', 
    'MQG.AX': 'Macquarie Group Ltd.', 
    'RIO.AX': 'Rio Tinto Ltd.', 
    'SCG.AX': 'Scentre Group', 
    'WPL.AX': 'Woodside Petroleum Ltd.', 
    'TCL.AX': 'Transurban Group', 
    'BXB.AX': 'Brambles Ltd.', 
    'WFD.AX': 'Westfield Corp.', 
    'AMC.AX': 'Amcor Ltd./Australia', 
    'SUN.AX': 'Suncorp Group Ltd.', 
    'QBE.AX': 'QBE Insurance Group Ltd.', 
    'NCM.AX': 'Newcrest Mining Ltd.', 
    'S32.AX': 'South32 Ltd.', 
    'AGL.AX': 'AGL Energy Ltd.', 
    'SYD.AX': 'Sydney Airport', 
    'AMP.AX': 'AMP Ltd.', 
    'IAG.AX': 'Insurance Australia Group Ltd.', 
    'GMG.AX': 'Goodman Group', 
    'ORG.AX': 'Origin Energy Ltd.', 
    'SGP.AX': 'Stockland', 
    'AZJ.AX': 'Aurizon Holdings Ltd.', 
    'FMG.AX': 'Fortescue Metals Group Ltd.', 
    'VCX.AX': 'Vicinity Centres', 
    'ASX.AX': 'ASX Ltd.', 
    'JHX.AX': 'James Hardie Industries plc', 
    'RHC.AX': 'Ramsay Health Care Ltd.', 
    'SHL.AX': 'Sonic Healthcare Ltd.', 
    'ALL.AX': 'Aristocrat Leisure Ltd.', 
    'APA.AX': 'APA Group', 
    'DXS.AX': 'Dexus Property Group', 
    'GPT.AX': 'GPT Group', 
    'TWE.AX': 'Treasury Wine Estates Ltd.', 
    'CTX.AX': 'Caltex Australia Ltd.', 
    'MGR.AX': 'Mirvac Group', 
    'OSH.AX': 'Oil Search Ltd.', 
    'LLC.AX': 'LendLease Group', 
    'MPL.AX': 'Medibank Pvt Ltd.', 
    'QAN.AX': 'Qantas Airways Ltd.', 
    'COH.AX': 'Cochlear Ltd.', 
    'ORI.AX': 'Orica Ltd.', 
    'STO.AX': 'Santos Ltd.', 
    'CGF.AX': 'Challenger Ltd./Australia', 
    'TTS.AX': 'Tatts Group Ltd.', 
    'CPU.AX': 'Computershare Ltd.', 
    'DUE.AX': 'DUET Group', 
    'BEN.AX': 'Bendigo & Adelaide Bank Ltd.', 
    'IPL.AX': 'Incitec Pivot Ltd.', 
    'BSL.AX': 'BlueScope Steel Ltd.', 
    'SEK.AX': 'SEEK Ltd.', 
    'CCL.AX': 'Coca-Cola Amatil Ltd.', 
    'RMD.AX': 'ResMed Inc.', 
    'CWN.AX': 'Crown Resorts Ltd.', 
    'DMP.AX': 'Dominos Pizza Enterprises Ltd.', 
    'BOQ.AX': 'Bank of Queensland Ltd.', 
    'SGR.AX': 'Star Entertainment Grp Ltd.', 
    'HSO.AX': 'Healthscope Ltd.', 
    'TAH.AX': 'Tabcorp Holdings Ltd.', 
    'AWC.AX': 'Alumina Ltd.', 
    'SKI.AX': 'Spark Infrastructure Group', 
    'BLD.AX': 'Boral Ltd.', 
    'ORA.AX': 'Orora Ltd.', 
    'ANN.AX': 'Ansell Ltd.', 
    'JBH.AX': 'JB Hi-Fi Ltd.', 
    'CYB.AX': 'CYBG plc', 
    'CIM.AX': 'CIMIC Group Ltd.', 
    'ALQ.AX': 'ALS Ltd.', 
    'MFG.AX': 'Magellan Financial Group Ltd.', 
    'QUB.AX': 'Qube Holdings Ltd.', 
    'HVN.AX': 'Harvey Norman Holdings Ltd.', 
    'LNK.AX': 'Link Administration Holdings Ltd.', 
    'IOF.AX': 'Investa Office Fund', 
    'VOC.AX': 'Vocus Communications Ltd.', 
    'ILU.AX': 'Iluka Resources Ltd.', 
    'AST.AX': 'AusNet Services', 
    'REA.AX': 'REA Group Ltd.', 
    'OZL.AX': 'OZ Minerals Ltd.', 
    'DOW.AX': 'Downer EDI Ltd.', 
    'INM.AX': 'Iron Mountain Inc.', 
    'CAR.AX': 'carsales.com Ltd.', 
    'IFL.AX': 'IOOF Holdings Ltd.', 
    'HGG.AX': 'Henderson Group plc', 
    'ABC.AX': 'Adelaide Brighton Ltd.', 
    'PPT.AX': 'Perpetual Ltd.', 
    'DLX.AX': 'DuluxGroup Ltd.', 
    'TPM.AX': 'TPG Telecom Ltd.', 
    'EVN.AX': 'Evolution Mining Ltd.', 
    'NST.AX': 'Northern Star Resources Ltd.', 
    'NHF.AX': 'nib holdings Ltd./Australia', 
    'MQA.AX': 'Macquarie Atlas Roads Group', 
    'CSR.AX': 'CSR Ltd.', 
    'MYX.AX': 'Mayne Pharma Group Ltd.', 
    'SGM.AX': 'Sims Metal Management Ltd.', 
    'MTS.AX': 'Metcash Ltd.', 
    'IGO.AX': 'Independence Group NL', 
    'MIN.AX': 'Mineral Resources Ltd.', 
    'FXJ.AX': 'Fairfax Media Ltd.', 
    'WOR.AX': 'WorleyParsons Ltd.', 
    'FLT.AX': 'Flight Centre Travel Group Ltd.', 
    'WHC.AX': 'Whitehaven Coal Ltd.', 
    'CHC.AX': 'Charter Hall Group', 
    'BTT.AX': 'BT Investment Management Ltd.', 
    'CWY.AX': 'Cleanaway Waste Management Ltd.', 
    'FBU.AX': 'Fletcher Building Ltd.', 
    'NVT.AX': 'Navitas Ltd.', 
    'GNC.AX': 'GrainCorp Ltd. Class A', 
    'SRX.AX': 'Sirtex Medical Ltd.', 
    'SDF.AX': 'Steadfast Group Ltd.', 
    'SCP.AX': 'Shopping Centres Australasia Property Group', 
    'NUF.AX': 'Nufarm Ltd./Australia', 
    'PRY.AX': 'Primary Health Care Ltd.', 
    'BKL.AX': 'Blackmores Ltd.', 
    'IRE.AX': 'IRESS Ltd.', 
    'BWP.AX': 'BWP Trust', 
    'IVC.AX': 'InvoCare Ltd.', 
    'RRL.AX': 'Regis Resources Ltd.', 
    'CQR.AX': 'Charter Hall Retail REIT', 
    'BAP.AX': 'Bapcor Ltd.', 
    'SUL.AX': 'Super Retail Group Ltd.', 
    'SIP.AX': 'Sigma Pharmaceuticals Ltd.', 
    'PMV.AX': 'Premier Investments Ltd.', 
    'CTD.AX': 'Corporate Travel Management Ltd.', 
    'AOG.AX': 'Aveo Group', 
    'TNE.AX': 'Technology One Ltd.', 
    'ARB.AX': 'ARB Corp. Ltd.', 
    'CMW.AX': 'Cromwell Property Group', 
    'GEM.AX': 'G8 Education Ltd.', 
    'PTM.AX': 'Platinum Asset Management Ltd.', 
    'BPT.AX': 'Beach Energy Ltd.', 
    'RWC.AX': 'Reliance Worldwide Corp. Ltd.', 
    'TME.AX': 'Trade Me Group Ltd.', 
    'PGH.AX': 'Pact Group Holdings Ltd.', 
    'SBM.AX': 'St. Barbara Ltd.', 
    'MYR.AX': 'Myer Holdings Ltd.', 
    'A2M.AX': 'a2 Milk Co. Ltd.', 
    'SPO.AX': 'Spotless Group Holdings Ltd.', 
    'SXL.AX': 'Southern Cross Media Group Ltd.', 
    'SAI.AX': 'SAI Global Ltd.', 
    'FPH.AX': 'Fisher & Paykel Healthcare Corp. Ltd.', 
    'AHG.AX': 'Automotive Holdings Group Ltd.', 
    'WEB.AX': 'Webjet Ltd.', 
    'MND.AX': 'Monadelphous Group Ltd.', 
    'BKW.AX': 'Brickworks Ltd.', 
    'ECX.AX': 'Eclipx Group Ltd.', 
    'ALU.AX': 'Altium Ltd.', 
    'AAD.AX': 'Ardent Leisure Group', 
    'ORE.AX': 'Orocobre Ltd.', 
    'MTR.AX': 'Mantra Group Ltd.', 
    'SFR.AX': 'Sandfire Resources NL', 
    'RSG.AX': 'Resolute Mining Ltd.', 
    'GUD.AX': 'GUD Holdings Ltd.', 
    'BAL.AX': 'Bellamys Australia Ltd.', 
    'NXT.AX': 'NEXTDC Ltd.', 
    'APO.AX': 'APN Outdoor Group Ltd.', 
    'CCP.AX': 'Credit Corp. Group Ltd.', 
    'SKC.AX': 'SKYCITY Entertainment Group Ltd.', 
    'ABP.AX': 'Abacus Property Group', 
    'NAN.AX': 'Nanosonics Ltd.', 
    'SPK.AX': 'Spark New Zealand Ltd.', 
    'MYO.AX': 'MYOB Group Ltd.', 
    'NWS.AX': 'News Corp.', 
    'SVW.AX': 'Seven Group Holdings Ltd.', 
    'CGC.AX': 'Costa Group Holdings Ltd.', 
    'BRG.AX': 'Breville Group Ltd.', 
    'GXY.AX': 'Galaxy Resources Ltd.', 
    'GWA.AX': 'GWA Group Ltd.', 
    'MMS.AX': 'McMillan Shakespeare Ltd.', 
    'SAR.AX': 'Saracen Mineral Holdings Ltd.', 
    'BGA.AX': 'Bega Cheese Ltd.', 
    'RFG.AX': 'Retail Food Group Ltd.', 
    'NSR.AX': 'National Storage REIT', 
    'GXL.AX': 'Greencross Ltd.', 
    'ACX.AX': 'Aconex Ltd.', 
    'WSA.AX': 'Western Areas Ltd.', 
    'API.AX': 'Australian Pharmaceutical Industries Ltd.', 
    'IFN.AX': 'Infigen Energy', 
    'IPH.AX': 'IPH Ltd.', 
    'SYR.AX': 'Syrah Resources Ltd.', 
    'GMA.AX': 'Genworth Mortgage Insurance Australia Ltd.', 
    'SKT.AX': 'SKY Network Television Ltd.', 
    'APN.AX': 'APN News & Media Ltd.', 
    'GTY.AX': 'Gateway Lifestyle', 
    'NEC.AX': 'Nine Entertainment Co. Holdings Ltd.', 
    'OML.AX': 'oOh!media Ltd.', 
    'FXL.AX': 'FlexiGroup Ltd./Australia', 
    'SWM.AX': 'Seven West Media Ltd.', 
    'PLS.AX': 'Pilbara Minerals Ltd.', 
    'GOZ.AX': 'Growthpoint Properties Australia Ltd.', 
    'PRU.AX': 'Perseus Mining Ltd.', 
    'HSN.AX': 'Hansen Technologies Ltd.', 
    'ASB.AX': 'Austal Ltd.', 
    'AHY.AX': 'Asaleo Care Ltd.', 
    'AAC.AX': 'Australian Agricultural Co. Ltd.', 
    'TGR.AX': 'Tassal Group Ltd.', 
    'AVN.AX': 'Aventus Retail Property Fund Ltd.', 
    'WTC.AX': 'WiseTech Global Ltd.', 
    'CNU.AX': 'Chorus Ltd.', 
    'REG.AX': 'Regis Healthcare Ltd.', 
    'FET.AX': 'Folkestone Education Trust', 
    'GOR.AX': 'Gold Road Resources Ltd.', 
    'ISD.AX': 'iSentia Group Ltd.', 
    'UGL.AX': 'UGL Ltd.', 
    'IEL.AX': 'IDP Education Ltd.', 
    'CKF.AX': 'Collins Foods Ltd.', 
    'TFC.AX': 'TFS Corp. Ltd.', 
    'RCG.AX': 'RCG Corp. Ltd.', 
    'CVO.AX': 'Cover-More Group Ltd.', 
    'EHE.AX': 'Estia Health Ltd.', 
    'JHC.AX': 'Japara Healthcare Ltd.', 
    'INA.AX': 'Ingenia Communities Group', 
    'VRT.AX': 'Virtus Health Ltd.', 
    'SIQ.AX': 'SmartGroup Corp. Ltd.', 
    'SHV.AX': 'Select Harvests Ltd.', 
    'SDA.AX': 'SpeedCast International Ltd.', 
    'GDI.AX': 'GDI Property Group', 
    'TIX.AX': '360 Capital Industrial Fund', 
    'BLA.AX': 'Blue Sky Alternative Investments Ltd.', 
    'ELD.AX': 'Elders Ltd.', 
    'ACK.AX': 'Auckland International Airport Ltd.', 
    'ARF.AX': 'Arena REIT', 
    'IPD.AX': 'ImpediMed Ltd.', 
    'KAR.AX': 'Karoon Gas Australia Ltd.', 
    'TOX.AX': 'Tox Free Solutions Ltd.', 
    'EML.AX': 'EML Payments Ltd.', 
    'MVF.AX': 'Monash IVF Group Ltd.', 
    'ISU.AX': 'iSelect Ltd.', 
    'OFX.AX': 'OFX Group Ltd.', 
    'CAB.AX': 'Cabcharge Australia Ltd.', 
    'SGF.AX': 'SG Fleet Group Ltd.', 
    'DCN.AX': 'Dacian Gold Ltd.', 
    'PRG.AX': 'Programmed Maintenance Services Ltd.', 
    'HPI.AX': 'Hotel Property Investments', 
    'AJA.AX': 'Astro Japan Property Group', 
    'FNP.AX': 'Freedom Foods Group Ltd.', 
    'VTG.AX': 'Vita Group Ltd.', 
    'OGC.AX': 'OceanaGold Corp.', 
    'BDR.AX': 'Beadell Resources Ltd.', 
    'AMA.AX': 'AMA Group Ltd.', 
    'VRL.AX': 'Village Roadshow Ltd.', 
    'RCR.AX': 'RCR Tomlinson Ltd.', 
    'SSM.AX': 'Service Stream Ltd.', 
    'EQT.AX': 'EQT Holdings Ltd.', 
    'AYS.AX': 'Amaysim Australia Ltd.', 
    'SLK.AX': 'SeaLink Travel Group Ltd.', 
    'BWX.AX': 'BWX Ltd.', 
    'WPP.AX': 'WPP AUNZ Ltd.', 
    'HFA.AX': 'HFA Holdings Ltd.', 
    'AWE.AX': 'AWE Ltd.', 
    'RFF.AX': 'Rural Funds Group', 
    'IDR.AX': 'Industria REIT', 
    'MSB.AX': 'Mesoblast Ltd.', 
    'CLW.AX': 'Charter Hall Long Wale REIT', 
    'LNG.AX': 'Liquefied Natural Gas Ltd.', 
    'MLX.AX': 'Metals X Ltd.', 
    'CL1.AX': 'Class Ltd.', 
    'RIC.AX': 'Ridley Corp. Ltd.', 
    'NTC.AX': 'NetComm Wireless Ltd.', 
    'IMF.AX': 'IMF Bentham Ltd.', 
    'KMD.AX': 'Kathmandu Holdings Ltd.', 
    'BBN.AX': 'Baby Bunting Group Ltd.', 
    'MLD.AX': 'MACA Ltd.', 
    'MNS.AX': 'Magnis Resources Ltd.', 
    'CWP.AX': 'Cedar Woods Properties Ltd.', 
    'CDD.AX': 'Cardno Ltd.', 
    'FAR.AX': 'FAR Ltd.', 
    'HFR.AX': 'Highfield Resources Ltd.', 
    'SPL.AX': 'Starpharma Holdings Ltd.', 
    'VIT.AX': 'Vitaco Holdings Ltd.', 
    'GHC.AX': 'Generation Healthcare REIT', 
    'TGA.AX': 'Thorn Group Ltd.', 
    'XPZ.AX': 'SPI 200 FUTURES   Dec16', 
    'VLW.AX': 'Villa World Ltd.', 
    'TRS.AX': 'Reject Shop Ltd.', 
    'LYC.AX': 'Lynas Corp. Ltd.', 
    'GBT.AX': 'GBST Holdings Ltd.', 
    'SXY.AX': 'Senex Energy Ltd.', 
    'VVR.AX': 'Viva Energy REIT', 
    'CSV.AX': 'CSG Ltd.', 
    'SIV.AX': 'Silver Chef Ltd.', 
    'MOC.AX': 'Mortgage Choice Ltd.', 
    'TEN.AX': 'Ten Network Holdings Ltd.', 
    'SEH.AX': 'Sino Gas & Energy Holdings Ltd.', 
    'IFM.AX': 'Infomedia Ltd.', 
    'WBA.AX': 'Webster Ltd.', 
    'AGI.AX': 'Ainsworth Game Technology Ltd.', 
    'EPW.AX': 'ERM Power Ltd.', 
    'MGC.AX': 'MG Unit Trust', 
    'EWC.AX': 'Energy World Corp. Ltd.', 
    'DRM.AX': 'Doray Minerals Ltd.', 
    'AQG.AX': 'Alacer Gold Corp.', 
    'FSF.AX': 'Fonterra Co-operative Group Ltd.', 
    'DNA.AX': 'Donaco International Ltd.', 
    'ADH.AX': 'Adairs Ltd.', 
    'AJX.AX': 'Alexium International Group Ltd.', 
    'PDN.AX': 'Paladin Energy Ltd.', 
    'CCV.AX': 'Cash Converters International Ltd.'}

symbols, names = np.array(list(symbol_dict.items())).T

quotes = [quotes_historical_yahoo_ochl(symbol, d1, d2, asobject=True)
          for symbol in symbols]

open = np.array([q.open for q in quotes]).astype(np.float)
close = np.array([q.close for q in quotes]).astype(np.float)

# The daily variations of the quotes are what carry most information
variation = close - open

###############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

###############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

###############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()
