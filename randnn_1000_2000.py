import tensorflow as tf
import random

log = open('randnn_1000_2000.log','w')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

x_train, x_test = x_train / 255.0, x_test / 255.0

num_classes = 10

x_train = tf.reshape(x_train, [-1, 784])
x_test = tf.reshape(x_test, [-1, 784])

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('Training started')

learning_rate = 0.001
num_epochs = 20
batch_size = 100

xavier = tf.keras.initializers.GlorotUniform()

### num_inputs, num_outputs, num_hiddennodes, num_edges
### 784 10 1000 2000
W = tf.Variable(xavier([2000]))
B = tf.Variable(tf.random.normal([1794]))
@tf.function
def node794(X):
    result = B[794] + tf.gather(X, 94, axis=1)*W[1902] + tf.gather(X, 266, axis=1)*W[1275] + tf.gather(X, 497, axis=1)*W[1970] + tf.gather(X, 709, axis=1)*W[1441] + node993(X)*W[1581]
    return(result)

@tf.function
def node795(X):
    result = B[795] + tf.gather(X, 513, axis=1)*W[1741] + tf.gather(X, 701, axis=1)*W[1236] + node1225(X)*W[1639]
    return(result)

@tf.function
def node796(X):
    result = B[796] + tf.gather(X, 414, axis=1)*W[93] + node1781(X)*W[834]
    return(result)

@tf.function
def node797(X):
    result = B[797] + tf.gather(X, 654, axis=1)*W[1748] + node820(X)*W[1119]
    return(result)

@tf.function
def node798(X):
    result = B[798] + tf.gather(X, 241, axis=1)*W[1124] + node1119(X)*W[483]
    return(result)

@tf.function
def node799(X):
    result = B[799] + tf.gather(X, 711, axis=1)*W[623] + tf.gather(X, 739, axis=1)*W[1135]
    return(result)

@tf.function
def node800(X):
    result = B[800] + node1545(X)*W[1470] + node1751(X)*W[1507]
    return(result)

@tf.function
def node801(X):
    result = B[801] + tf.gather(X, 222, axis=1)*W[1038] + tf.gather(X, 478, axis=1)*W[1348] + node1004(X)*W[1196] + node1379(X)*W[574] + node1409(X)*W[1966]
    return(result)

@tf.function
def node802(X):
    result = B[802] + node1335(X)*W[585] + node1630(X)*W[344]
    return(result)

@tf.function
def node803(X):
    result = B[803]
    return(result)

@tf.function
def node804(X):
    result = B[804]
    return(result)

@tf.function
def node805(X):
    result = B[805]
    return(result)

@tf.function
def node806(X):
    result = B[806] + node1045(X)*W[1986] + node1365(X)*W[1045] + node1499(X)*W[334]
    return(result)

@tf.function
def node807(X):
    result = B[807] + node909(X)*W[1620]
    return(result)

@tf.function
def node808(X):
    result = B[808] + tf.gather(X, 635, axis=1)*W[1351] + node837(X)*W[1753] + node1449(X)*W[653] + node1739(X)*W[357]
    return(result)

@tf.function
def node809(X):
    result = B[809]
    return(result)

@tf.function
def node810(X):
    result = B[810] + tf.gather(X, 435, axis=1)*W[1845]
    return(result)

@tf.function
def node811(X):
    result = B[811] + node1520(X)*W[1555]
    return(result)

@tf.function
def node812(X):
    result = B[812] + node1420(X)*W[1286]
    return(result)

@tf.function
def node813(X):
    result = B[813] + tf.gather(X, 58, axis=1)*W[1325] + node1131(X)*W[1869]
    return(result)

@tf.function
def node814(X):
    result = B[814] + node1073(X)*W[1428]
    return(result)

@tf.function
def node815(X):
    result = B[815] + tf.gather(X, 242, axis=1)*W[1918] + tf.gather(X, 531, axis=1)*W[1281]
    return(result)

@tf.function
def node816(X):
    result = B[816] + node982(X)*W[1147] + node1297(X)*W[1462]
    return(result)

@tf.function
def node817(X):
    result = B[817] + tf.gather(X, 710, axis=1)*W[1645] + node872(X)*W[1780] + node1535(X)*W[1823]
    return(result)

@tf.function
def node818(X):
    result = B[818] + tf.gather(X, 69, axis=1)*W[1221] + node1195(X)*W[439]
    return(result)

@tf.function
def node819(X):
    result = B[819]
    return(result)

@tf.function
def node820(X):
    result = B[820] + tf.gather(X, 315, axis=1)*W[416] + tf.gather(X, 327, axis=1)*W[813] + node1788(X)*W[1315]
    return(result)

@tf.function
def node821(X):
    result = B[821] + tf.gather(X, 74, axis=1)*W[964]
    return(result)

@tf.function
def node822(X):
    result = B[822] + node1196(X)*W[1175]
    return(result)

@tf.function
def node823(X):
    result = B[823] + tf.gather(X, 552, axis=1)*W[1190] + node943(X)*W[99] + node1719(X)*W[629]
    return(result)

@tf.function
def node824(X):
    result = B[824] + tf.gather(X, 512, axis=1)*W[1408] + node1304(X)*W[152] + node1467(X)*W[985] + node1574(X)*W[875]
    return(result)

@tf.function
def node825(X):
    result = B[825] + tf.gather(X, 172, axis=1)*W[752] + node1449(X)*W[1760]
    return(result)

@tf.function
def node826(X):
    result = B[826] + tf.gather(X, 533, axis=1)*W[1086] + tf.gather(X, 584, axis=1)*W[1336]
    return(result)

@tf.function
def node827(X):
    result = B[827] + tf.gather(X, 82, axis=1)*W[1867] + tf.gather(X, 475, axis=1)*W[102] + tf.gather(X, 772, axis=1)*W[1761] + node817(X)*W[141] + node1300(X)*W[1435] + node1639(X)*W[1673]
    return(result)

@tf.function
def node828(X):
    result = B[828] + tf.gather(X, 299, axis=1)*W[247] + node1037(X)*W[253] + node1277(X)*W[1983]
    return(result)

@tf.function
def node829(X):
    result = B[829] + tf.gather(X, 310, axis=1)*W[1895]
    return(result)

@tf.function
def node830(X):
    result = B[830]
    return(result)

@tf.function
def node831(X):
    result = B[831] + node830(X)*W[1558] + node1679(X)*W[1510]
    return(result)

@tf.function
def node832(X):
    result = B[832] + tf.gather(X, 234, axis=1)*W[1749]
    return(result)

@tf.function
def node833(X):
    result = B[833] + tf.gather(X, 363, axis=1)*W[1004] + node1426(X)*W[1665]
    return(result)

@tf.function
def node834(X):
    result = B[834] + tf.gather(X, 54, axis=1)*W[853] + tf.gather(X, 265, axis=1)*W[923] + tf.gather(X, 735, axis=1)*W[1349] + node1355(X)*W[199]
    return(result)

@tf.function
def node835(X):
    result = B[835] + tf.gather(X, 381, axis=1)*W[812] + node918(X)*W[240] + node1122(X)*W[1571] + node1138(X)*W[1002] + node1725(X)*W[630]
    return(result)

@tf.function
def node836(X):
    result = B[836] + tf.gather(X, 451, axis=1)*W[1566]
    return(result)

@tf.function
def node837(X):
    result = B[837] + tf.gather(X, 230, axis=1)*W[68] + node1438(X)*W[972] + node1607(X)*W[1158]
    return(result)

@tf.function
def node838(X):
    result = B[838] + tf.gather(X, 649, axis=1)*W[1328] + tf.gather(X, 726, axis=1)*W[1822] + node914(X)*W[993] + node1391(X)*W[644] + node1598(X)*W[816] + node1625(X)*W[1568]
    return(result)

@tf.function
def node839(X):
    result = B[839] + node1009(X)*W[75]
    return(result)

@tf.function
def node840(X):
    result = B[840]
    return(result)

@tf.function
def node841(X):
    result = B[841] + tf.gather(X, 275, axis=1)*W[1886]
    return(result)

@tf.function
def node842(X):
    result = B[842] + node1758(X)*W[544]
    return(result)

@tf.function
def node843(X):
    result = B[843] + node884(X)*W[1613] + node1047(X)*W[978]
    return(result)

@tf.function
def node844(X):
    result = B[844] + tf.gather(X, 138, axis=1)*W[1439] + tf.gather(X, 141, axis=1)*W[265] + tf.gather(X, 674, axis=1)*W[724] + node1541(X)*W[618]
    return(result)

@tf.function
def node845(X):
    result = B[845] + node1163(X)*W[992] + node1620(X)*W[1264]
    return(result)

@tf.function
def node846(X):
    result = B[846]
    return(result)

@tf.function
def node847(X):
    result = B[847]
    return(result)

@tf.function
def node848(X):
    result = B[848] + tf.gather(X, 67, axis=1)*W[1762] + node1278(X)*W[666]
    return(result)

@tf.function
def node849(X):
    result = B[849] + node1752(X)*W[461]
    return(result)

@tf.function
def node850(X):
    result = B[850] + node1727(X)*W[1091]
    return(result)

@tf.function
def node851(X):
    result = B[851] + tf.gather(X, 720, axis=1)*W[1800]
    return(result)

@tf.function
def node852(X):
    result = B[852] + tf.gather(X, 218, axis=1)*W[438] + node1397(X)*W[777] + node1792(X)*W[1502]
    return(result)

@tf.function
def node853(X):
    result = B[853] + tf.gather(X, 20, axis=1)*W[299] + tf.gather(X, 122, axis=1)*W[351] + tf.gather(X, 415, axis=1)*W[91] + tf.gather(X, 595, axis=1)*W[900] + node1436(X)*W[944] + node1472(X)*W[1060]
    return(result)

@tf.function
def node854(X):
    result = B[854] + tf.gather(X, 288, axis=1)*W[1025] + tf.gather(X, 684, axis=1)*W[27] + node1079(X)*W[1065]
    return(result)

@tf.function
def node855(X):
    result = B[855] + tf.gather(X, 566, axis=1)*W[1456] + node1586(X)*W[625]
    return(result)

@tf.function
def node856(X):
    result = B[856] + tf.gather(X, 245, axis=1)*W[1857] + node1671(X)*W[695]
    return(result)

@tf.function
def node857(X):
    result = B[857] + node1443(X)*W[817]
    return(result)

@tf.function
def node858(X):
    result = B[858] + tf.gather(X, 103, axis=1)*W[1424] + tf.gather(X, 211, axis=1)*W[458] + tf.gather(X, 727, axis=1)*W[1783] + node1604(X)*W[411]
    return(result)

@tf.function
def node859(X):
    result = B[859] + node1679(X)*W[1757]
    return(result)

@tf.function
def node860(X):
    result = B[860] + node1420(X)*W[313]
    return(result)

@tf.function
def node861(X):
    result = B[861] + node954(X)*W[1454] + node1633(X)*W[385]
    return(result)

@tf.function
def node862(X):
    result = B[862] + tf.gather(X, 281, axis=1)*W[1897] + tf.gather(X, 621, axis=1)*W[1460] + node948(X)*W[1674] + node1211(X)*W[228] + node1322(X)*W[1968]
    return(result)

@tf.function
def node863(X):
    result = B[863] + tf.gather(X, 538, axis=1)*W[271] + tf.gather(X, 733, axis=1)*W[551]
    return(result)

@tf.function
def node864(X):
    result = B[864] + node934(X)*W[321]
    return(result)

@tf.function
def node865(X):
    result = B[865] + tf.gather(X, 407, axis=1)*W[1242] + node1000(X)*W[394] + node1172(X)*W[86]
    return(result)

@tf.function
def node866(X):
    result = B[866] + tf.gather(X, 758, axis=1)*W[1784] + node1284(X)*W[117] + node1588(X)*W[401]
    return(result)

@tf.function
def node867(X):
    result = B[867] + tf.gather(X, 343, axis=1)*W[1553] + tf.gather(X, 674, axis=1)*W[316] + node1516(X)*W[227]
    return(result)

@tf.function
def node868(X):
    result = B[868] + tf.gather(X, 639, axis=1)*W[319] + tf.gather(X, 644, axis=1)*W[839]
    return(result)

@tf.function
def node869(X):
    result = B[869] + tf.gather(X, 762, axis=1)*W[1794]
    return(result)

@tf.function
def node870(X):
    result = B[870] + node949(X)*W[13] + node1517(X)*W[302] + node1742(X)*W[926]
    return(result)

@tf.function
def node871(X):
    result = B[871] + tf.gather(X, 148, axis=1)*W[1251]
    return(result)

@tf.function
def node872(X):
    result = B[872] + tf.gather(X, 11, axis=1)*W[1885] + tf.gather(X, 711, axis=1)*W[1215] + node867(X)*W[603] + node1619(X)*W[528]
    return(result)

@tf.function
def node873(X):
    result = B[873] + node1306(X)*W[105]
    return(result)

@tf.function
def node874(X):
    result = B[874] + node908(X)*W[307] + node1286(X)*W[1390]
    return(result)

@tf.function
def node875(X):
    result = B[875] + tf.gather(X, 358, axis=1)*W[1343] + node854(X)*W[743]
    return(result)

@tf.function
def node876(X):
    result = B[876]
    return(result)

@tf.function
def node877(X):
    result = B[877]
    return(result)

@tf.function
def node878(X):
    result = B[878] + tf.gather(X, 557, axis=1)*W[885] + node1378(X)*W[1280]
    return(result)

@tf.function
def node879(X):
    result = B[879] + tf.gather(X, 644, axis=1)*W[1493] + node1210(X)*W[286] + node1739(X)*W[991]
    return(result)

@tf.function
def node880(X):
    result = B[880] + tf.gather(X, 48, axis=1)*W[558] + tf.gather(X, 87, axis=1)*W[22]
    return(result)

@tf.function
def node881(X):
    result = B[881] + tf.gather(X, 123, axis=1)*W[1563]
    return(result)

@tf.function
def node882(X):
    result = B[882] + node1062(X)*W[1707]
    return(result)

@tf.function
def node883(X):
    result = B[883] + tf.gather(X, 123, axis=1)*W[974] + node1498(X)*W[1203] + node1651(X)*W[642]
    return(result)

@tf.function
def node884(X):
    result = B[884] + tf.gather(X, 432, axis=1)*W[1840] + node1262(X)*W[1069]
    return(result)

@tf.function
def node885(X):
    result = B[885] + node1005(X)*W[1163]
    return(result)

@tf.function
def node886(X):
    result = B[886] + node1189(X)*W[375]
    return(result)

@tf.function
def node887(X):
    result = B[887] + tf.gather(X, 344, axis=1)*W[1623] + tf.gather(X, 780, axis=1)*W[1041] + node1323(X)*W[1152] + node1525(X)*W[222]
    return(result)

@tf.function
def node888(X):
    result = B[888] + tf.gather(X, 214, axis=1)*W[1026]
    return(result)

@tf.function
def node889(X):
    result = B[889] + tf.gather(X, 342, axis=1)*W[589]
    return(result)

@tf.function
def node890(X):
    result = B[890] + tf.gather(X, 610, axis=1)*W[1772] + tf.gather(X, 694, axis=1)*W[1363] + node1205(X)*W[624]
    return(result)

@tf.function
def node891(X):
    result = B[891] + tf.gather(X, 24, axis=1)*W[346] + tf.gather(X, 524, axis=1)*W[1015] + node926(X)*W[855]
    return(result)

@tf.function
def node892(X):
    result = B[892] + tf.gather(X, 95, axis=1)*W[431] + tf.gather(X, 275, axis=1)*W[1926] + tf.gather(X, 476, axis=1)*W[1552]
    return(result)

@tf.function
def node893(X):
    result = B[893] + tf.gather(X, 324, axis=1)*W[717] + node1709(X)*W[1584]
    return(result)

@tf.function
def node894(X):
    result = B[894] + node877(X)*W[1154]
    return(result)

@tf.function
def node895(X):
    result = B[895] + node1486(X)*W[1628]
    return(result)

@tf.function
def node896(X):
    result = B[896] + tf.gather(X, 285, axis=1)*W[427] + tf.gather(X, 436, axis=1)*W[1217]
    return(result)

@tf.function
def node897(X):
    result = B[897] + tf.gather(X, 53, axis=1)*W[159] + node1737(X)*W[1669]
    return(result)

@tf.function
def node898(X):
    result = B[898] + tf.gather(X, 189, axis=1)*W[1808] + tf.gather(X, 604, axis=1)*W[767] + node846(X)*W[481] + node1643(X)*W[925]
    return(result)

@tf.function
def node899(X):
    result = B[899]
    return(result)

@tf.function
def node900(X):
    result = B[900] + node1126(X)*W[627]
    return(result)

@tf.function
def node901(X):
    result = B[901] + node1338(X)*W[80] + node1524(X)*W[891]
    return(result)

@tf.function
def node902(X):
    result = B[902] + node1143(X)*W[47] + node1302(X)*W[901]
    return(result)

@tf.function
def node903(X):
    result = B[903] + tf.gather(X, 348, axis=1)*W[400] + tf.gather(X, 478, axis=1)*W[90] + node1760(X)*W[328]
    return(result)

@tf.function
def node904(X):
    result = B[904] + tf.gather(X, 336, axis=1)*W[651] + tf.gather(X, 499, axis=1)*W[391] + node910(X)*W[1567] + node1012(X)*W[1928]
    return(result)

@tf.function
def node905(X):
    result = B[905] + node1600(X)*W[1569]
    return(result)

@tf.function
def node906(X):
    result = B[906] + tf.gather(X, 182, axis=1)*W[856] + tf.gather(X, 249, axis=1)*W[1747] + tf.gather(X, 554, axis=1)*W[1371] + tf.gather(X, 661, axis=1)*W[164] + node1223(X)*W[694]
    return(result)

@tf.function
def node907(X):
    result = B[907] + node983(X)*W[916]
    return(result)

@tf.function
def node908(X):
    result = B[908] + tf.gather(X, 452, axis=1)*W[335]
    return(result)

@tf.function
def node909(X):
    result = B[909]
    return(result)

@tf.function
def node910(X):
    result = B[910] + tf.gather(X, 146, axis=1)*W[383] + tf.gather(X, 236, axis=1)*W[1172]
    return(result)

@tf.function
def node911(X):
    result = B[911] + node829(X)*W[980] + node995(X)*W[940] + node1510(X)*W[730]
    return(result)

@tf.function
def node912(X):
    result = B[912] + node1158(X)*W[1854] + node1272(X)*W[250] + node1758(X)*W[1482]
    return(result)

@tf.function
def node913(X):
    result = B[913] + tf.gather(X, 240, axis=1)*W[696] + tf.gather(X, 568, axis=1)*W[1561] + tf.gather(X, 701, axis=1)*W[959] + node1002(X)*W[195] + node1481(X)*W[1085]
    return(result)

@tf.function
def node914(X):
    result = B[914] + node1058(X)*W[414]
    return(result)

@tf.function
def node915(X):
    result = B[915] + tf.gather(X, 626, axis=1)*W[986] + node1214(X)*W[1167]
    return(result)

@tf.function
def node916(X):
    result = B[916] + tf.gather(X, 115, axis=1)*W[1361] + tf.gather(X, 132, axis=1)*W[1684]
    return(result)

@tf.function
def node917(X):
    result = B[917] + tf.gather(X, 492, axis=1)*W[1294] + tf.gather(X, 609, axis=1)*W[1487] + node858(X)*W[209]
    return(result)

@tf.function
def node918(X):
    result = B[918] + tf.gather(X, 27, axis=1)*W[1059] + node889(X)*W[889] + node1781(X)*W[1938]
    return(result)

@tf.function
def node919(X):
    result = B[919] + tf.gather(X, 547, axis=1)*W[533] + node1635(X)*W[1601]
    return(result)

@tf.function
def node920(X):
    result = B[920] + tf.gather(X, 589, axis=1)*W[1594] + node983(X)*W[1880] + node1294(X)*W[1208] + node1374(X)*W[1020] + node1721(X)*W[230]
    return(result)

@tf.function
def node921(X):
    result = B[921] + tf.gather(X, 409, axis=1)*W[1486]
    return(result)

@tf.function
def node922(X):
    result = B[922]
    return(result)

@tf.function
def node923(X):
    result = B[923] + node1099(X)*W[821] + node1438(X)*W[1649]
    return(result)

@tf.function
def node924(X):
    result = B[924] + tf.gather(X, 550, axis=1)*W[1817]
    return(result)

@tf.function
def node925(X):
    result = B[925] + node1170(X)*W[1540] + node1645(X)*W[547]
    return(result)

@tf.function
def node926(X):
    result = B[926] + tf.gather(X, 160, axis=1)*W[1382]
    return(result)

@tf.function
def node927(X):
    result = B[927] + tf.gather(X, 479, axis=1)*W[61] + node1108(X)*W[1519]
    return(result)

@tf.function
def node928(X):
    result = B[928]
    return(result)

@tf.function
def node929(X):
    result = B[929] + node822(X)*W[1782] + node1127(X)*W[1887] + node1218(X)*W[1802] + node1474(X)*W[303]
    return(result)

@tf.function
def node930(X):
    result = B[930] + node1250(X)*W[1256] + node1725(X)*W[329]
    return(result)

@tf.function
def node931(X):
    result = B[931] + tf.gather(X, 64, axis=1)*W[699]
    return(result)

@tf.function
def node932(X):
    result = B[932] + tf.gather(X, 186, axis=1)*W[791] + tf.gather(X, 192, axis=1)*W[1615] + tf.gather(X, 205, axis=1)*W[1330] + node1155(X)*W[194]
    return(result)

@tf.function
def node933(X):
    result = B[933] + node1611(X)*W[1576]
    return(result)

@tf.function
def node934(X):
    result = B[934] + node1718(X)*W[1307]
    return(result)

@tf.function
def node935(X):
    result = B[935] + node1583(X)*W[1078] + node1598(X)*W[378]
    return(result)

@tf.function
def node936(X):
    result = B[936] + node1361(X)*W[1477]
    return(result)

@tf.function
def node937(X):
    result = B[937] + tf.gather(X, 680, axis=1)*W[797] + node811(X)*W[279] + node1018(X)*W[1766]
    return(result)

@tf.function
def node938(X):
    result = B[938] + tf.gather(X, 449, axis=1)*W[1745] + node1072(X)*W[1715] + node1492(X)*W[434]
    return(result)

@tf.function
def node939(X):
    result = B[939] + node1367(X)*W[43] + node1759(X)*W[268]
    return(result)

@tf.function
def node940(X):
    result = B[940] + tf.gather(X, 537, axis=1)*W[294] + tf.gather(X, 722, axis=1)*W[56]
    return(result)

@tf.function
def node941(X):
    result = B[941] + tf.gather(X, 203, axis=1)*W[1161]
    return(result)

@tf.function
def node942(X):
    result = B[942] + tf.gather(X, 711, axis=1)*W[284]
    return(result)

@tf.function
def node943(X):
    result = B[943] + tf.gather(X, 17, axis=1)*W[596] + tf.gather(X, 599, axis=1)*W[1821]
    return(result)

@tf.function
def node944(X):
    result = B[944] + tf.gather(X, 55, axis=1)*W[1833] + node1125(X)*W[69] + node1315(X)*W[1409] + node1379(X)*W[1940]
    return(result)

@tf.function
def node945(X):
    result = B[945] + node1680(X)*W[1644] + node1766(X)*W[1700]
    return(result)

@tf.function
def node946(X):
    result = B[946] + tf.gather(X, 139, axis=1)*W[484] + tf.gather(X, 563, axis=1)*W[578] + node969(X)*W[941]
    return(result)

@tf.function
def node947(X):
    result = B[947]
    return(result)

@tf.function
def node948(X):
    result = B[948] + tf.gather(X, 124, axis=1)*W[1702] + tf.gather(X, 542, axis=1)*W[1304] + tf.gather(X, 646, axis=1)*W[1909] + node1237(X)*W[1832] + node1637(X)*W[200]
    return(result)

@tf.function
def node949(X):
    result = B[949] + tf.gather(X, 102, axis=1)*W[611] + tf.gather(X, 628, axis=1)*W[1042] + node1273(X)*W[1618]
    return(result)

@tf.function
def node950(X):
    result = B[950] + tf.gather(X, 119, axis=1)*W[1340] + node1558(X)*W[979]
    return(result)

@tf.function
def node951(X):
    result = B[951] + tf.gather(X, 527, axis=1)*W[727] + node1010(X)*W[123] + node1346(X)*W[324] + node1741(X)*W[433]
    return(result)

@tf.function
def node952(X):
    result = B[952] + tf.gather(X, 105, axis=1)*W[478] + tf.gather(X, 120, axis=1)*W[1668] + node1258(X)*W[1399] + node1287(X)*W[166]
    return(result)

@tf.function
def node953(X):
    result = B[953] + tf.gather(X, 422, axis=1)*W[243] + tf.gather(X, 574, axis=1)*W[1699] + tf.gather(X, 680, axis=1)*W[1159] + tf.gather(X, 773, axis=1)*W[997] + node1282(X)*W[748]
    return(result)

@tf.function
def node954(X):
    result = B[954] + tf.gather(X, 347, axis=1)*W[1140] + node1212(X)*W[211]
    return(result)

@tf.function
def node955(X):
    result = B[955] + tf.gather(X, 677, axis=1)*W[1513]
    return(result)

@tf.function
def node956(X):
    result = B[956] + node1605(X)*W[1901]
    return(result)

@tf.function
def node957(X):
    result = B[957] + tf.gather(X, 268, axis=1)*W[1120] + node933(X)*W[773] + node1251(X)*W[674]
    return(result)

@tf.function
def node958(X):
    result = B[958] + tf.gather(X, 151, axis=1)*W[1816] + tf.gather(X, 403, axis=1)*W[820]
    return(result)

@tf.function
def node959(X):
    result = B[959] + node924(X)*W[965] + node1639(X)*W[697]
    return(result)

@tf.function
def node960(X):
    result = B[960]
    return(result)

@tf.function
def node961(X):
    result = B[961] + node1199(X)*W[1169] + node1486(X)*W[1849] + node1508(X)*W[559]
    return(result)

@tf.function
def node962(X):
    result = B[962]
    return(result)

@tf.function
def node963(X):
    result = B[963] + tf.gather(X, 215, axis=1)*W[451]
    return(result)

@tf.function
def node964(X):
    result = B[964] + node1336(X)*W[432] + node1373(X)*W[1580]
    return(result)

@tf.function
def node965(X):
    result = B[965] + tf.gather(X, 51, axis=1)*W[1925]
    return(result)

@tf.function
def node966(X):
    result = B[966]
    return(result)

@tf.function
def node967(X):
    result = B[967] + tf.gather(X, 764, axis=1)*W[217] + node996(X)*W[688] + node1543(X)*W[269]
    return(result)

@tf.function
def node968(X):
    result = B[968] + node1349(X)*W[235] + node1403(X)*W[868]
    return(result)

@tf.function
def node969(X):
    result = B[969] + node861(X)*W[124] + node905(X)*W[952] + node1784(X)*W[1732]
    return(result)

@tf.function
def node970(X):
    result = B[970] + node1429(X)*W[1550]
    return(result)

@tf.function
def node971(X):
    result = B[971] + tf.gather(X, 353, axis=1)*W[428] + node1302(X)*W[1611]
    return(result)

@tf.function
def node972(X):
    result = B[972] + node1069(X)*W[457] + node1621(X)*W[778]
    return(result)

@tf.function
def node973(X):
    result = B[973] + node1600(X)*W[1916]
    return(result)

@tf.function
def node974(X):
    result = B[974]
    return(result)

@tf.function
def node975(X):
    result = B[975] + tf.gather(X, 126, axis=1)*W[63] + tf.gather(X, 678, axis=1)*W[1905]
    return(result)

@tf.function
def node976(X):
    result = B[976] + tf.gather(X, 19, axis=1)*W[1178] + node966(X)*W[1108] + node1768(X)*W[1014]
    return(result)

@tf.function
def node977(X):
    result = B[977]
    return(result)

@tf.function
def node978(X):
    result = B[978] + node935(X)*W[264] + node1496(X)*W[14] + node1549(X)*W[41] + node1745(X)*W[1166]
    return(result)

@tf.function
def node979(X):
    result = B[979]
    return(result)

@tf.function
def node980(X):
    result = B[980] + tf.gather(X, 413, axis=1)*W[127] + tf.gather(X, 499, axis=1)*W[1379]
    return(result)

@tf.function
def node981(X):
    result = B[981] + node1224(X)*W[1496]
    return(result)

@tf.function
def node982(X):
    result = B[982] + tf.gather(X, 498, axis=1)*W[554] + node1251(X)*W[1009] + node1443(X)*W[650]
    return(result)

@tf.function
def node983(X):
    result = B[983] + tf.gather(X, 161, axis=1)*W[1054] + node1759(X)*W[1830]
    return(result)

@tf.function
def node984(X):
    result = B[984]
    return(result)

@tf.function
def node985(X):
    result = B[985] + tf.gather(X, 8, axis=1)*W[1421] + tf.gather(X, 433, axis=1)*W[1530] + node1722(X)*W[1943]
    return(result)

@tf.function
def node986(X):
    result = B[986] + node1523(X)*W[1185]
    return(result)

@tf.function
def node987(X):
    result = B[987] + node979(X)*W[884]
    return(result)

@tf.function
def node988(X):
    result = B[988]
    return(result)

@tf.function
def node989(X):
    result = B[989] + tf.gather(X, 756, axis=1)*W[1910] + node1133(X)*W[1468] + node1688(X)*W[718]
    return(result)

@tf.function
def node990(X):
    result = B[990] + tf.gather(X, 253, axis=1)*W[296]
    return(result)

@tf.function
def node991(X):
    result = B[991]
    return(result)

@tf.function
def node992(X):
    result = B[992] + node1762(X)*W[1788]
    return(result)

@tf.function
def node993(X):
    result = B[993]
    return(result)

@tf.function
def node994(X):
    result = B[994] + node1086(X)*W[1915]
    return(result)

@tf.function
def node995(X):
    result = B[995] + tf.gather(X, 274, axis=1)*W[420] + tf.gather(X, 494, axis=1)*W[1621] + tf.gather(X, 582, axis=1)*W[1971]
    return(result)

@tf.function
def node996(X):
    result = B[996] + tf.gather(X, 108, axis=1)*W[938] + node1679(X)*W[1162]
    return(result)

@tf.function
def node997(X):
    result = B[997] + node830(X)*W[1779] + node1342(X)*W[1844] + node1726(X)*W[1383]
    return(result)

@tf.function
def node998(X):
    result = B[998] + tf.gather(X, 350, axis=1)*W[1472]
    return(result)

@tf.function
def node999(X):
    result = B[999] + tf.gather(X, 548, axis=1)*W[62] + node1642(X)*W[1252]
    return(result)

@tf.function
def node1000(X):
    result = B[1000] + node1164(X)*W[1528]
    return(result)

@tf.function
def node1001(X):
    result = B[1001] + node824(X)*W[1133] + node1496(X)*W[241]
    return(result)

@tf.function
def node1002(X):
    result = B[1002]
    return(result)

@tf.function
def node1003(X):
    result = B[1003] + node1339(X)*W[935] + node1459(X)*W[1942]
    return(result)

@tf.function
def node1004(X):
    result = B[1004] + node1234(X)*W[1807]
    return(result)

@tf.function
def node1005(X):
    result = B[1005] + node1412(X)*W[864]
    return(result)

@tf.function
def node1006(X):
    result = B[1006] + node1058(X)*W[148]
    return(result)

@tf.function
def node1007(X):
    result = B[1007]
    return(result)

@tf.function
def node1008(X):
    result = B[1008] + tf.gather(X, 80, axis=1)*W[705] + tf.gather(X, 451, axis=1)*W[136] + tf.gather(X, 456, axis=1)*W[919] + node1661(X)*W[495]
    return(result)

@tf.function
def node1009(X):
    result = B[1009]
    return(result)

@tf.function
def node1010(X):
    result = B[1010] + tf.gather(X, 312, axis=1)*W[331] + tf.gather(X, 354, axis=1)*W[1156] + node1017(X)*W[1738] + node1498(X)*W[1590]
    return(result)

@tf.function
def node1011(X):
    result = B[1011] + tf.gather(X, 522, axis=1)*W[111] + node936(X)*W[1090]
    return(result)

@tf.function
def node1012(X):
    result = B[1012] + tf.gather(X, 188, axis=1)*W[1650] + tf.gather(X, 193, axis=1)*W[874] + tf.gather(X, 688, axis=1)*W[1898] + node871(X)*W[1265]
    return(result)

@tf.function
def node1013(X):
    result = B[1013] + tf.gather(X, 638, axis=1)*W[1255] + tf.gather(X, 774, axis=1)*W[915] + node1227(X)*W[1385] + node1365(X)*W[1422] + node1703(X)*W[616]
    return(result)

@tf.function
def node1014(X):
    result = B[1014] + tf.gather(X, 267, axis=1)*W[1032] + node1288(X)*W[442]
    return(result)

@tf.function
def node1015(X):
    result = B[1015] + tf.gather(X, 542, axis=1)*W[403] + tf.gather(X, 631, axis=1)*W[1101] + tf.gather(X, 709, axis=1)*W[1733] + node990(X)*W[1081]
    return(result)

@tf.function
def node1016(X):
    result = B[1016] + tf.gather(X, 159, axis=1)*W[187] + tf.gather(X, 301, axis=1)*W[171] + node860(X)*W[57] + node1467(X)*W[336]
    return(result)

@tf.function
def node1017(X):
    result = B[1017] + tf.gather(X, 24, axis=1)*W[712] + tf.gather(X, 47, axis=1)*W[1321] + node880(X)*W[934] + node1754(X)*W[1283]
    return(result)

@tf.function
def node1018(X):
    result = B[1018] + node1601(X)*W[1646] + node1678(X)*W[519]
    return(result)

@tf.function
def node1019(X):
    result = B[1019] + node1008(X)*W[1339]
    return(result)

@tf.function
def node1020(X):
    result = B[1020] + node1157(X)*W[314]
    return(result)

@tf.function
def node1021(X):
    result = B[1021] + node1348(X)*W[505]
    return(result)

@tf.function
def node1022(X):
    result = B[1022] + tf.gather(X, 1, axis=1)*W[1237] + tf.gather(X, 51, axis=1)*W[1150] + tf.gather(X, 706, axis=1)*W[1447] + node1688(X)*W[162]
    return(result)

@tf.function
def node1023(X):
    result = B[1023] + tf.gather(X, 763, axis=1)*W[1937] + node900(X)*W[1923]
    return(result)

@tf.function
def node1024(X):
    result = B[1024] + tf.gather(X, 64, axis=1)*W[1583] + tf.gather(X, 424, axis=1)*W[1419] + tf.gather(X, 478, axis=1)*W[996] + tf.gather(X, 630, axis=1)*W[361] + node798(X)*W[809] + node800(X)*W[126] + node1617(X)*W[356]
    return(result)

@tf.function
def node1025(X):
    result = B[1025] + tf.gather(X, 657, axis=1)*W[129] + node839(X)*W[1919] + node1021(X)*W[586] + node1663(X)*W[798] + node1776(X)*W[257]
    return(result)

@tf.function
def node1026(X):
    result = B[1026] + tf.gather(X, 395, axis=1)*W[1685] + tf.gather(X, 654, axis=1)*W[710] + node1162(X)*W[702] + node1714(X)*W[1787]
    return(result)

@tf.function
def node1027(X):
    result = B[1027] + tf.gather(X, 478, axis=1)*W[1589] + node999(X)*W[1137]
    return(result)

@tf.function
def node1028(X):
    result = B[1028] + tf.gather(X, 184, axis=1)*W[1099]
    return(result)

@tf.function
def node1029(X):
    result = B[1029] + tf.gather(X, 61, axis=1)*W[1614] + tf.gather(X, 471, axis=1)*W[708]
    return(result)

@tf.function
def node1030(X):
    result = B[1030] + tf.gather(X, 45, axis=1)*W[598]
    return(result)

@tf.function
def node1031(X):
    result = B[1031]
    return(result)

@tf.function
def node1032(X):
    result = B[1032] + node832(X)*W[1899]
    return(result)

@tf.function
def node1033(X):
    result = B[1033]
    return(result)

@tf.function
def node1034(X):
    result = B[1034] + node828(X)*W[720] + node1104(X)*W[1936]
    return(result)

@tf.function
def node1035(X):
    result = B[1035] + tf.gather(X, 720, axis=1)*W[580] + node1267(X)*W[977] + node1713(X)*W[143] + node1778(X)*W[758]
    return(result)

@tf.function
def node1036(X):
    result = B[1036]
    return(result)

@tf.function
def node1037(X):
    result = B[1037] + tf.gather(X, 618, axis=1)*W[413]
    return(result)

@tf.function
def node1038(X):
    result = B[1038] + tf.gather(X, 546, axis=1)*W[1245]
    return(result)

@tf.function
def node1039(X):
    result = B[1039] + tf.gather(X, 22, axis=1)*W[805] + tf.gather(X, 165, axis=1)*W[392] + tf.gather(X, 263, axis=1)*W[1592] + tf.gather(X, 620, axis=1)*W[600] + node874(X)*W[1653]
    return(result)

@tf.function
def node1040(X):
    result = B[1040] + tf.gather(X, 255, axis=1)*W[1872] + tf.gather(X, 373, axis=1)*W[883] + node1074(X)*W[202] + node1144(X)*W[1848]
    return(result)

@tf.function
def node1041(X):
    result = B[1041] + node953(X)*W[1941]
    return(result)

@tf.function
def node1042(X):
    result = B[1042] + tf.gather(X, 372, axis=1)*W[1295] + tf.gather(X, 517, axis=1)*W[1171] + node999(X)*W[909] + node1277(X)*W[1411]
    return(result)

@tf.function
def node1043(X):
    result = B[1043] + node1289(X)*W[681]
    return(result)

@tf.function
def node1044(X):
    result = B[1044]
    return(result)

@tf.function
def node1045(X):
    result = B[1045] + tf.gather(X, 254, axis=1)*W[1008] + node1292(X)*W[590]
    return(result)

@tf.function
def node1046(X):
    result = B[1046] + tf.gather(X, 42, axis=1)*W[1612]
    return(result)

@tf.function
def node1047(X):
    result = B[1047] + node1638(X)*W[150]
    return(result)

@tf.function
def node1048(X):
    result = B[1048] + tf.gather(X, 42, axis=1)*W[133] + tf.gather(X, 43, axis=1)*W[1318] + tf.gather(X, 724, axis=1)*W[907] + node961(X)*W[967] + node1351(X)*W[1384]
    return(result)

@tf.function
def node1049(X):
    result = B[1049] + tf.gather(X, 474, axis=1)*W[1070] + node1163(X)*W[961] + node1401(X)*W[550]
    return(result)

@tf.function
def node1050(X):
    result = B[1050] + tf.gather(X, 624, axis=1)*W[371] + node1761(X)*W[376]
    return(result)

@tf.function
def node1051(X):
    result = B[1051] + node1181(X)*W[1586] + node1190(X)*W[168]
    return(result)

@tf.function
def node1052(X):
    result = B[1052] + node1407(X)*W[734]
    return(result)

@tf.function
def node1053(X):
    result = B[1053] + tf.gather(X, 616, axis=1)*W[1995]
    return(result)

@tf.function
def node1054(X):
    result = B[1054] + tf.gather(X, 502, axis=1)*W[1080] + node1083(X)*W[684] + node1513(X)*W[1922]
    return(result)

@tf.function
def node1055(X):
    result = B[1055] + tf.gather(X, 201, axis=1)*W[1498] + node1396(X)*W[1876]
    return(result)

@tf.function
def node1056(X):
    result = B[1056] + node1155(X)*W[1727]
    return(result)

@tf.function
def node1057(X):
    result = B[1057] + node857(X)*W[827] + node980(X)*W[1975] + node1762(X)*W[1736]
    return(result)

@tf.function
def node1058(X):
    result = B[1058] + tf.gather(X, 276, axis=1)*W[1993]
    return(result)

@tf.function
def node1059(X):
    result = B[1059]
    return(result)

@tf.function
def node1060(X):
    result = B[1060] + node1639(X)*W[323]
    return(result)

@tf.function
def node1061(X):
    result = B[1061] + tf.gather(X, 630, axis=1)*W[1030]
    return(result)

@tf.function
def node1062(X):
    result = B[1062] + tf.gather(X, 318, axis=1)*W[1181] + tf.gather(X, 523, axis=1)*W[1635] + tf.gather(X, 633, axis=1)*W[789] + node1595(X)*W[755]
    return(result)

@tf.function
def node1063(X):
    result = B[1063] + tf.gather(X, 208, axis=1)*W[1648] + node1173(X)*W[1258] + node1489(X)*W[236]
    return(result)

@tf.function
def node1064(X):
    result = B[1064] + tf.gather(X, 687, axis=1)*W[1791] + node1520(X)*W[1625]
    return(result)

@tf.function
def node1065(X):
    result = B[1065] + tf.gather(X, 478, axis=1)*W[922] + node974(X)*W[121] + node1711(X)*W[785]
    return(result)

@tf.function
def node1066(X):
    result = B[1066] + tf.gather(X, 29, axis=1)*W[1564] + tf.gather(X, 336, axis=1)*W[1290] + tf.gather(X, 451, axis=1)*W[675] + node1629(X)*W[1480]
    return(result)

@tf.function
def node1067(X):
    result = B[1067] + node1313(X)*W[745]
    return(result)

@tf.function
def node1068(X):
    result = B[1068] + tf.gather(X, 487, axis=1)*W[67] + node1774(X)*W[516]
    return(result)

@tf.function
def node1069(X):
    result = B[1069] + node823(X)*W[386] + node899(X)*W[593] + node1375(X)*W[1739]
    return(result)

@tf.function
def node1070(X):
    result = B[1070] + tf.gather(X, 297, axis=1)*W[46]
    return(result)

@tf.function
def node1071(X):
    result = B[1071] + tf.gather(X, 623, axis=1)*W[1303]
    return(result)

@tf.function
def node1072(X):
    result = B[1072] + tf.gather(X, 238, axis=1)*W[1322]
    return(result)

@tf.function
def node1073(X):
    result = B[1073] + tf.gather(X, 87, axis=1)*W[501]
    return(result)

@tf.function
def node1074(X):
    result = B[1074] + node1000(X)*W[310]
    return(result)

@tf.function
def node1075(X):
    result = B[1075] + tf.gather(X, 492, axis=1)*W[448] + node801(X)*W[1544]
    return(result)

@tf.function
def node1076(X):
    result = B[1076] + tf.gather(X, 291, axis=1)*W[858] + tf.gather(X, 757, axis=1)*W[664] + node890(X)*W[364]
    return(result)

@tf.function
def node1077(X):
    result = B[1077] + tf.gather(X, 342, axis=1)*W[904] + tf.gather(X, 755, axis=1)*W[525] + node1262(X)*W[373]
    return(result)

@tf.function
def node1078(X):
    result = B[1078] + tf.gather(X, 670, axis=1)*W[850] + node930(X)*W[910]
    return(result)

@tf.function
def node1079(X):
    result = B[1079] + node1469(X)*W[1820]
    return(result)

@tf.function
def node1080(X):
    result = B[1080] + tf.gather(X, 640, axis=1)*W[486] + node872(X)*W[1570]
    return(result)

@tf.function
def node1081(X):
    result = B[1081] + node1612(X)*W[1838]
    return(result)

@tf.function
def node1082(X):
    result = B[1082] + tf.gather(X, 82, axis=1)*W[775] + tf.gather(X, 667, axis=1)*W[1168] + node1137(X)*W[1917] + node1725(X)*W[768]
    return(result)

@tf.function
def node1083(X):
    result = B[1083]
    return(result)

@tf.function
def node1084(X):
    result = B[1084]
    return(result)

@tf.function
def node1085(X):
    result = B[1085] + tf.gather(X, 447, axis=1)*W[106] + node799(X)*W[927]
    return(result)

@tf.function
def node1086(X):
    result = B[1086] + tf.gather(X, 699, axis=1)*W[1803] + node1205(X)*W[987] + node1619(X)*W[620]
    return(result)

@tf.function
def node1087(X):
    result = B[1087]
    return(result)

@tf.function
def node1088(X):
    result = B[1088]
    return(result)

@tf.function
def node1089(X):
    result = B[1089] + tf.gather(X, 56, axis=1)*W[1602] + tf.gather(X, 60, axis=1)*W[1533] + tf.gather(X, 390, axis=1)*W[683] + tf.gather(X, 548, axis=1)*W[1019] + node1680(X)*W[1074]
    return(result)

@tf.function
def node1090(X):
    result = B[1090] + tf.gather(X, 216, axis=1)*W[1088]
    return(result)

@tf.function
def node1091(X):
    result = B[1091] + node1005(X)*W[1241] + node1148(X)*W[1138] + node1673(X)*W[917]
    return(result)

@tf.function
def node1092(X):
    result = B[1092] + tf.gather(X, 641, axis=1)*W[523] + node1131(X)*W[291]
    return(result)

@tf.function
def node1093(X):
    result = B[1093] + tf.gather(X, 750, axis=1)*W[40]
    return(result)

@tf.function
def node1094(X):
    result = B[1094] + tf.gather(X, 474, axis=1)*W[845] + tf.gather(X, 507, axis=1)*W[1367] + tf.gather(X, 521, axis=1)*W[847] + tf.gather(X, 748, axis=1)*W[1342] + node874(X)*W[796] + node1384(X)*W[1523]
    return(result)

@tf.function
def node1095(X):
    result = B[1095] + tf.gather(X, 668, axis=1)*W[1003]
    return(result)

@tf.function
def node1096(X):
    result = B[1096]
    return(result)

@tf.function
def node1097(X):
    result = B[1097] + tf.gather(X, 7, axis=1)*W[350] + tf.gather(X, 205, axis=1)*W[524] + tf.gather(X, 263, axis=1)*W[504] + node895(X)*W[1912] + node1416(X)*W[1981] + node1627(X)*W[1999]
    return(result)

@tf.function
def node1098(X):
    result = B[1098] + tf.gather(X, 177, axis=1)*W[892] + tf.gather(X, 274, axis=1)*W[409] + tf.gather(X, 514, axis=1)*W[1976] + node947(X)*W[1989]
    return(result)

@tf.function
def node1099(X):
    result = B[1099] + tf.gather(X, 579, axis=1)*W[300] + tf.gather(X, 680, axis=1)*W[270] + node1318(X)*W[1358] + node1560(X)*W[1582]
    return(result)

@tf.function
def node1100(X):
    result = B[1100] + tf.gather(X, 701, axis=1)*W[1149]
    return(result)

@tf.function
def node1101(X):
    result = B[1101] + tf.gather(X, 534, axis=1)*W[1775] + tf.gather(X, 560, axis=1)*W[835] + node1496(X)*W[1430]
    return(result)

@tf.function
def node1102(X):
    result = B[1102] + tf.gather(X, 85, axis=1)*W[808] + tf.gather(X, 780, axis=1)*W[1375]
    return(result)

@tf.function
def node1103(X):
    result = B[1103]
    return(result)

@tf.function
def node1104(X):
    result = B[1104] + tf.gather(X, 117, axis=1)*W[605] + node1376(X)*W[1306]
    return(result)

@tf.function
def node1105(X):
    result = B[1105] + node1371(X)*W[1955]
    return(result)

@tf.function
def node1106(X):
    result = B[1106] + node1091(X)*W[1051]
    return(result)

@tf.function
def node1107(X):
    result = B[1107] + node1419(X)*W[532]
    return(result)

@tf.function
def node1108(X):
    result = B[1108] + tf.gather(X, 89, axis=1)*W[867] + node1783(X)*W[854]
    return(result)

@tf.function
def node1109(X):
    result = B[1109] + tf.gather(X, 547, axis=1)*W[422]
    return(result)

@tf.function
def node1110(X):
    result = B[1110] + tf.gather(X, 135, axis=1)*W[412] + node1374(X)*W[838]
    return(result)

@tf.function
def node1111(X):
    result = B[1111] + tf.gather(X, 444, axis=1)*W[423] + tf.gather(X, 605, axis=1)*W[693]
    return(result)

@tf.function
def node1112(X):
    result = B[1112]
    return(result)

@tf.function
def node1113(X):
    result = B[1113] + node1272(X)*W[1683]
    return(result)

@tf.function
def node1114(X):
    result = B[1114] + tf.gather(X, 504, axis=1)*W[1341] + node957(X)*W[212]
    return(result)

@tf.function
def node1115(X):
    result = B[1115] + node915(X)*W[450] + node1358(X)*W[1758]
    return(result)

@tf.function
def node1116(X):
    result = B[1116] + tf.gather(X, 5, axis=1)*W[377] + tf.gather(X, 534, axis=1)*W[440] + node890(X)*W[1222] + node1543(X)*W[531]
    return(result)

@tf.function
def node1117(X):
    result = B[1117] + tf.gather(X, 242, axis=1)*W[949] + tf.gather(X, 758, axis=1)*W[234]
    return(result)

@tf.function
def node1118(X):
    result = B[1118] + tf.gather(X, 422, axis=1)*W[896] + node939(X)*W[1364] + node1323(X)*W[38]
    return(result)

@tf.function
def node1119(X):
    result = B[1119] + tf.gather(X, 660, axis=1)*W[508] + node892(X)*W[753]
    return(result)

@tf.function
def node1120(X):
    result = B[1120] + tf.gather(X, 25, axis=1)*W[39] + node1135(X)*W[877]
    return(result)

@tf.function
def node1121(X):
    result = B[1121] + tf.gather(X, 512, axis=1)*W[165] + node1197(X)*W[1889] + node1712(X)*W[1720]
    return(result)

@tf.function
def node1122(X):
    result = B[1122] + node816(X)*W[931] + node862(X)*W[1977] + node964(X)*W[1948]
    return(result)

@tf.function
def node1123(X):
    result = B[1123] + node815(X)*W[207] + node1559(X)*W[1633]
    return(result)

@tf.function
def node1124(X):
    result = B[1124] + node1636(X)*W[1226]
    return(result)

@tf.function
def node1125(X):
    result = B[1125] + tf.gather(X, 129, axis=1)*W[315] + tf.gather(X, 421, axis=1)*W[396] + tf.gather(X, 759, axis=1)*W[1881]
    return(result)

@tf.function
def node1126(X):
    result = B[1126] + tf.gather(X, 70, axis=1)*W[1688] + tf.gather(X, 127, axis=1)*W[347] + tf.gather(X, 702, axis=1)*W[824]
    return(result)

@tf.function
def node1127(X):
    result = B[1127] + tf.gather(X, 274, axis=1)*W[1771] + node864(X)*W[526] + node1320(X)*W[759] + node1627(X)*W[895]
    return(result)

@tf.function
def node1128(X):
    result = B[1128] + node1059(X)*W[849]
    return(result)

@tf.function
def node1129(X):
    result = B[1129] + tf.gather(X, 332, axis=1)*W[536]
    return(result)

@tf.function
def node1130(X):
    result = B[1130] + tf.gather(X, 366, axis=1)*W[1027] + node1095(X)*W[1426] + node1234(X)*W[1436]
    return(result)

@tf.function
def node1131(X):
    result = B[1131] + node1283(X)*W[232] + node1791(X)*W[1853]
    return(result)

@tf.function
def node1132(X):
    result = B[1132]
    return(result)

@tf.function
def node1133(X):
    result = B[1133]
    return(result)

@tf.function
def node1134(X):
    result = B[1134] + tf.gather(X, 157, axis=1)*W[1664] + tf.gather(X, 324, axis=1)*W[1953] + node1364(X)*W[762]
    return(result)

@tf.function
def node1135(X):
    result = B[1135] + tf.gather(X, 723, axis=1)*W[1184] + node1775(X)*W[471]
    return(result)

@tf.function
def node1136(X):
    result = B[1136] + node885(X)*W[330] + node1446(X)*W[449]
    return(result)

@tf.function
def node1137(X):
    result = B[1137] + node1496(X)*W[1146]
    return(result)

@tf.function
def node1138(X):
    result = B[1138]
    return(result)

@tf.function
def node1139(X):
    result = B[1139] + tf.gather(X, 550, axis=1)*W[238]
    return(result)

@tf.function
def node1140(X):
    result = B[1140]
    return(result)

@tf.function
def node1141(X):
    result = B[1141] + node1253(X)*W[632]
    return(result)

@tf.function
def node1142(X):
    result = B[1142] + tf.gather(X, 689, axis=1)*W[638] + node956(X)*W[886]
    return(result)

@tf.function
def node1143(X):
    result = B[1143] + node1513(X)*W[114]
    return(result)

@tf.function
def node1144(X):
    result = B[1144] + tf.gather(X, 537, axis=1)*W[1157] + tf.gather(X, 581, axis=1)*W[1173] + tf.gather(X, 598, axis=1)*W[1717] + node835(X)*W[333] + node1133(X)*W[1965]
    return(result)

@tf.function
def node1145(X):
    result = B[1145] + tf.gather(X, 417, axis=1)*W[1141]
    return(result)

@tf.function
def node1146(X):
    result = B[1146] + node1493(X)*W[223]
    return(result)

@tf.function
def node1147(X):
    result = B[1147] + tf.gather(X, 655, axis=1)*W[1891]
    return(result)

@tf.function
def node1148(X):
    result = B[1148] + node819(X)*W[518]
    return(result)

@tf.function
def node1149(X):
    result = B[1149]
    return(result)

@tf.function
def node1150(X):
    result = B[1150] + tf.gather(X, 567, axis=1)*W[95] + node898(X)*W[1182] + node1142(X)*W[882] + node1507(X)*W[459] + node1694(X)*W[399]
    return(result)

@tf.function
def node1151(X):
    result = B[1151] + tf.gather(X, 260, axis=1)*W[287] + tf.gather(X, 664, axis=1)*W[1270] + node998(X)*W[1247] + node1087(X)*W[842] + node1239(X)*W[561] + node1518(X)*W[769]
    return(result)

@tf.function
def node1152(X):
    result = B[1152]
    return(result)

@tf.function
def node1153(X):
    result = B[1153] + tf.gather(X, 206, axis=1)*W[290]
    return(result)

@tf.function
def node1154(X):
    result = B[1154] + tf.gather(X, 540, axis=1)*W[975]
    return(result)

@tf.function
def node1155(X):
    result = B[1155]
    return(result)

@tf.function
def node1156(X):
    result = B[1156] + tf.gather(X, 61, axis=1)*W[1403] + node1629(X)*W[1298]
    return(result)

@tf.function
def node1157(X):
    result = B[1157]
    return(result)

@tf.function
def node1158(X):
    result = B[1158] + tf.gather(X, 583, axis=1)*W[245] + node1125(X)*W[1253] + node1687(X)*W[1864]
    return(result)

@tf.function
def node1159(X):
    result = B[1159] + tf.gather(X, 745, axis=1)*W[843] + node853(X)*W[1338] + node961(X)*W[1094] + node1095(X)*W[1207]
    return(result)

@tf.function
def node1160(X):
    result = B[1160] + node858(X)*W[266] + node1363(X)*W[538]
    return(result)

@tf.function
def node1161(X):
    result = B[1161] + node905(X)*W[1497] + node1685(X)*W[120]
    return(result)

@tf.function
def node1162(X):
    result = B[1162] + tf.gather(X, 240, axis=1)*W[474] + tf.gather(X, 443, axis=1)*W[301] + tf.gather(X, 597, axis=1)*W[1388]
    return(result)

@tf.function
def node1163(X):
    result = B[1163] + node798(X)*W[1961] + node1189(X)*W[1372] + node1276(X)*W[1954]
    return(result)

@tf.function
def node1164(X):
    result = B[1164] + tf.gather(X, 219, axis=1)*W[277] + node1178(X)*W[363]
    return(result)

@tf.function
def node1165(X):
    result = B[1165] + node1012(X)*W[1579] + node1693(X)*W[1366] + node1747(X)*W[920]
    return(result)

@tf.function
def node1166(X):
    result = B[1166] + tf.gather(X, 225, axis=1)*W[610] + node821(X)*W[1662] + node1251(X)*W[285] + node1681(X)*W[1951]
    return(result)

@tf.function
def node1167(X):
    result = B[1167] + tf.gather(X, 40, axis=1)*W[1957] + node1121(X)*W[818] + node1449(X)*W[1812]
    return(result)

@tf.function
def node1168(X):
    result = B[1168] + tf.gather(X, 200, axis=1)*W[87] + node1028(X)*W[1538]
    return(result)

@tf.function
def node1169(X):
    result = B[1169]
    return(result)

@tf.function
def node1170(X):
    result = B[1170] + tf.gather(X, 604, axis=1)*W[28] + tf.gather(X, 623, axis=1)*W[149] + node1769(X)*W[1350]
    return(result)

@tf.function
def node1171(X):
    result = B[1171] + tf.gather(X, 225, axis=1)*W[579] + node1182(X)*W[1314] + node1472(X)*W[1284] + node1741(X)*W[176]
    return(result)

@tf.function
def node1172(X):
    result = B[1172] + tf.gather(X, 598, axis=1)*W[1504] + node870(X)*W[852]
    return(result)

@tf.function
def node1173(X):
    result = B[1173] + tf.gather(X, 168, axis=1)*W[192] + node1279(X)*W[678] + node1516(X)*W[1939]
    return(result)

@tf.function
def node1174(X):
    result = B[1174] + tf.gather(X, 440, axis=1)*W[635] + tf.gather(X, 606, axis=1)*W[10] + node1654(X)*W[84]
    return(result)

@tf.function
def node1175(X):
    result = B[1175] + tf.gather(X, 730, axis=1)*W[185] + node1111(X)*W[1326]
    return(result)

@tf.function
def node1176(X):
    result = B[1176]
    return(result)

@tf.function
def node1177(X):
    result = B[1177] + tf.gather(X, 312, axis=1)*W[1046] + tf.gather(X, 411, axis=1)*W[1839] + tf.gather(X, 693, axis=1)*W[1125] + node930(X)*W[242] + node1424(X)*W[1431]
    return(result)

@tf.function
def node1178(X):
    result = B[1178] + tf.gather(X, 291, axis=1)*W[1769] + node857(X)*W[1127] + node1065(X)*W[1224]
    return(result)

@tf.function
def node1179(X):
    result = B[1179]
    return(result)

@tf.function
def node1180(X):
    result = B[1180] + tf.gather(X, 437, axis=1)*W[1377]
    return(result)

@tf.function
def node1181(X):
    result = B[1181] + tf.gather(X, 582, axis=1)*W[108] + node1305(X)*W[646]
    return(result)

@tf.function
def node1182(X):
    result = B[1182] + node972(X)*W[744]
    return(result)

@tf.function
def node1183(X):
    result = B[1183] + tf.gather(X, 634, axis=1)*W[1374]
    return(result)

@tf.function
def node1184(X):
    result = B[1184] + tf.gather(X, 778, axis=1)*W[1389] + node1643(X)*W[1205]
    return(result)

@tf.function
def node1185(X):
    result = B[1185] + tf.gather(X, 18, axis=1)*W[201] + tf.gather(X, 225, axis=1)*W[930] + tf.gather(X, 347, axis=1)*W[1694] + tf.gather(X, 392, axis=1)*W[898] + node1128(X)*W[1331] + node1250(X)*W[1187]
    return(result)

@tf.function
def node1186(X):
    result = B[1186] + node1045(X)*W[564] + node1199(X)*W[1417] + node1309(X)*W[1107]
    return(result)

@tf.function
def node1187(X):
    result = B[1187] + tf.gather(X, 188, axis=1)*W[421] + node815(X)*W[1012] + node931(X)*W[1682]
    return(result)

@tf.function
def node1188(X):
    result = B[1188] + node968(X)*W[861] + node998(X)*W[475] + node1387(X)*W[540] + node1474(X)*W[1183]
    return(result)

@tf.function
def node1189(X):
    result = B[1189] + node1414(X)*W[35]
    return(result)

@tf.function
def node1190(X):
    result = B[1190] + tf.gather(X, 703, axis=1)*W[581]
    return(result)

@tf.function
def node1191(X):
    result = B[1191]
    return(result)

@tf.function
def node1192(X):
    result = B[1192] + tf.gather(X, 414, axis=1)*W[144] + node971(X)*W[1116]
    return(result)

@tf.function
def node1193(X):
    result = B[1193] + tf.gather(X, 7, axis=1)*W[1254] + tf.gather(X, 298, axis=1)*W[640]
    return(result)

@tf.function
def node1194(X):
    result = B[1194] + tf.gather(X, 537, axis=1)*W[381] + node1530(X)*W[1693]
    return(result)

@tf.function
def node1195(X):
    result = B[1195] + tf.gather(X, 386, axis=1)*W[951] + node1591(X)*W[444]
    return(result)

@tf.function
def node1196(X):
    result = B[1196] + node1772(X)*W[1863]
    return(result)

@tf.function
def node1197(X):
    result = B[1197] + tf.gather(X, 325, axis=1)*W[819] + node1079(X)*W[637] + node1449(X)*W[576]
    return(result)

@tf.function
def node1198(X):
    result = B[1198] + tf.gather(X, 366, axis=1)*W[1515] + tf.gather(X, 602, axis=1)*W[1508]
    return(result)

@tf.function
def node1199(X):
    result = B[1199] + tf.gather(X, 531, axis=1)*W[1034] + tf.gather(X, 591, axis=1)*W[932] + node1630(X)*W[20]
    return(result)

@tf.function
def node1200(X):
    result = B[1200] + tf.gather(X, 704, axis=1)*W[686]
    return(result)

@tf.function
def node1201(X):
    result = B[1201] + tf.gather(X, 128, axis=1)*W[1934] + node849(X)*W[929] + node1149(X)*W[1539] + node1315(X)*W[1214]
    return(result)

@tf.function
def node1202(X):
    result = B[1202]
    return(result)

@tf.function
def node1203(X):
    result = B[1203] + tf.gather(X, 654, axis=1)*W[1814]
    return(result)

@tf.function
def node1204(X):
    result = B[1204] + node1347(X)*W[1873]
    return(result)

@tf.function
def node1205(X):
    result = B[1205] + tf.gather(X, 131, axis=1)*W[1017] + node1130(X)*W[21]
    return(result)

@tf.function
def node1206(X):
    result = B[1206] + tf.gather(X, 589, axis=1)*W[1786] + tf.gather(X, 613, axis=1)*W[1461] + node1044(X)*W[415] + node1226(X)*W[216] + node1775(X)*W[685]
    return(result)

@tf.function
def node1207(X):
    result = B[1207] + tf.gather(X, 179, axis=1)*W[830] + tf.gather(X, 697, axis=1)*W[1333] + node951(X)*W[615] + node1306(X)*W[116]
    return(result)

@tf.function
def node1208(X):
    result = B[1208] + node1283(X)*W[1344]
    return(result)

@tf.function
def node1209(X):
    result = B[1209] + node897(X)*W[779] + node1206(X)*W[1410] + node1633(X)*W[1092]
    return(result)

@tf.function
def node1210(X):
    result = B[1210] + tf.gather(X, 425, axis=1)*W[609]
    return(result)

@tf.function
def node1211(X):
    result = B[1211] + tf.gather(X, 675, axis=1)*W[1072]
    return(result)

@tf.function
def node1212(X):
    result = B[1212] + node1425(X)*W[1476] + node1428(X)*W[465] + node1713(X)*W[869]
    return(result)

@tf.function
def node1213(X):
    result = B[1213] + tf.gather(X, 417, axis=1)*W[1616] + tf.gather(X, 465, axis=1)*W[1229] + tf.gather(X, 518, axis=1)*W[592] + node1686(X)*W[1115]
    return(result)

@tf.function
def node1214(X):
    result = B[1214]
    return(result)

@tf.function
def node1215(X):
    result = B[1215]
    return(result)

@tf.function
def node1216(X):
    result = B[1216] + node1464(X)*W[1959]
    return(result)

@tf.function
def node1217(X):
    result = B[1217] + node1398(X)*W[1525] + node1611(X)*W[1827]
    return(result)

@tf.function
def node1218(X):
    result = B[1218] + node962(X)*W[1979] + node1652(X)*W[437]
    return(result)

@tf.function
def node1219(X):
    result = B[1219] + tf.gather(X, 567, axis=1)*W[177] + node893(X)*W[462]
    return(result)

@tf.function
def node1220(X):
    result = B[1220] + tf.gather(X, 373, axis=1)*W[1893] + node1190(X)*W[1950] + node1533(X)*W[1846]
    return(result)

@tf.function
def node1221(X):
    result = B[1221] + tf.gather(X, 637, axis=1)*W[1608]
    return(result)

@tf.function
def node1222(X):
    result = B[1222] + tf.gather(X, 21, axis=1)*W[1516] + node1434(X)*W[1276] + node1736(X)*W[53]
    return(result)

@tf.function
def node1223(X):
    result = B[1223] + tf.gather(X, 350, axis=1)*W[496] + node1102(X)*W[175]
    return(result)

@tf.function
def node1224(X):
    result = B[1224] + tf.gather(X, 270, axis=1)*W[147] + node1257(X)*W[722] + node1320(X)*W[1643] + node1337(X)*W[454]
    return(result)

@tf.function
def node1225(X):
    result = B[1225] + tf.gather(X, 481, axis=1)*W[946]
    return(result)

@tf.function
def node1226(X):
    result = B[1226] + tf.gather(X, 207, axis=1)*W[1062] + tf.gather(X, 276, axis=1)*W[387] + tf.gather(X, 438, axis=1)*W[772] + tf.gather(X, 692, axis=1)*W[792] + node814(X)*W[1005] + node1382(X)*W[1210] + node1484(X)*W[322]
    return(result)

@tf.function
def node1227(X):
    result = B[1227] + node1128(X)*W[1073] + node1174(X)*W[1053]
    return(result)

@tf.function
def node1228(X):
    result = B[1228] + node1792(X)*W[511]
    return(result)

@tf.function
def node1229(X):
    result = B[1229] + tf.gather(X, 348, axis=1)*W[1991] + tf.gather(X, 611, axis=1)*W[1287]
    return(result)

@tf.function
def node1230(X):
    result = B[1230] + node1109(X)*W[1921] + node1788(X)*W[65]
    return(result)

@tf.function
def node1231(X):
    result = B[1231] + node1006(X)*W[1177] + node1668(X)*W[1815]
    return(result)

@tf.function
def node1232(X):
    result = B[1232] + tf.gather(X, 492, axis=1)*W[1145]
    return(result)

@tf.function
def node1233(X):
    result = B[1233] + tf.gather(X, 94, axis=1)*W[1246] + node1674(X)*W[1202]
    return(result)

@tf.function
def node1234(X):
    result = B[1234] + node918(X)*W[1467] + node1006(X)*W[1765] + node1345(X)*W[488]
    return(result)

@tf.function
def node1235(X):
    result = B[1235] + node988(X)*W[1532]
    return(result)

@tf.function
def node1236(X):
    result = B[1236] + tf.gather(X, 621, axis=1)*W[560]
    return(result)

@tf.function
def node1237(X):
    result = B[1237] + tf.gather(X, 671, axis=1)*W[1128] + node975(X)*W[970] + node1267(X)*W[489] + node1550(X)*W[1952]
    return(result)

@tf.function
def node1238(X):
    result = B[1238] + tf.gather(X, 737, axis=1)*W[1929]
    return(result)

@tf.function
def node1239(X):
    result = B[1239] + tf.gather(X, 445, axis=1)*W[1474] + tf.gather(X, 477, axis=1)*W[1911] + node1209(X)*W[1634] + node1280(X)*W[85] + node1770(X)*W[1605]
    return(result)

@tf.function
def node1240(X):
    result = B[1240] + tf.gather(X, 81, axis=1)*W[220] + tf.gather(X, 671, axis=1)*W[19] + tf.gather(X, 711, axis=1)*W[225] + node1129(X)*W[793]
    return(result)

@tf.function
def node1241(X):
    result = B[1241] + tf.gather(X, 560, axis=1)*W[844] + tf.gather(X, 660, axis=1)*W[1292] + tf.gather(X, 779, axis=1)*W[1194] + node1537(X)*W[160]
    return(result)

@tf.function
def node1242(X):
    result = B[1242]
    return(result)

@tf.function
def node1243(X):
    result = B[1243] + tf.gather(X, 215, axis=1)*W[1309] + tf.gather(X, 580, axis=1)*W[1198]
    return(result)

@tf.function
def node1244(X):
    result = B[1244] + node1336(X)*W[1471]
    return(result)

@tf.function
def node1245(X):
    result = B[1245] + tf.gather(X, 116, axis=1)*W[1520] + tf.gather(X, 124, axis=1)*W[33]
    return(result)

@tf.function
def node1246(X):
    result = B[1246] + tf.gather(X, 754, axis=1)*W[1591] + node896(X)*W[34]
    return(result)

@tf.function
def node1247(X):
    result = B[1247] + tf.gather(X, 728, axis=1)*W[251] + tf.gather(X, 780, axis=1)*W[360] + node1038(X)*W[1267] + node1453(X)*W[1963] + node1726(X)*W[1746]
    return(result)

@tf.function
def node1248(X):
    result = B[1248] + tf.gather(X, 269, axis=1)*W[751] + node806(X)*W[1354]
    return(result)

@tf.function
def node1249(X):
    result = B[1249] + tf.gather(X, 723, axis=1)*W[527] + node910(X)*W[719]
    return(result)

@tf.function
def node1250(X):
    result = B[1250] + tf.gather(X, 43, axis=1)*W[1040] + node1200(X)*W[631] + node1323(X)*W[513] + node1746(X)*W[783]
    return(result)

@tf.function
def node1251(X):
    result = B[1251] + tf.gather(X, 253, axis=1)*W[443] + tf.gather(X, 657, axis=1)*W[1560] + node860(X)*W[771]
    return(result)

@tf.function
def node1252(X):
    result = B[1252] + tf.gather(X, 290, axis=1)*W[42] + tf.gather(X, 726, axis=1)*W[417]
    return(result)

@tf.function
def node1253(X):
    result = B[1253] + tf.gather(X, 371, axis=1)*W[973] + tf.gather(X, 764, axis=1)*W[537] + node1438(X)*W[79]
    return(result)

@tf.function
def node1254(X):
    result = B[1254]
    return(result)

@tf.function
def node1255(X):
    result = B[1255] + tf.gather(X, 383, axis=1)*W[1018] + node1215(X)*W[1767]
    return(result)

@tf.function
def node1256(X):
    result = B[1256] + tf.gather(X, 102, axis=1)*W[304] + tf.gather(X, 487, axis=1)*W[1742] + node934(X)*W[1407] + node1142(X)*W[1809]
    return(result)

@tf.function
def node1257(X):
    result = B[1257] + tf.gather(X, 505, axis=1)*W[976] + node1587(X)*W[196]
    return(result)

@tf.function
def node1258(X):
    result = B[1258] + tf.gather(X, 711, axis=1)*W[539] + node1247(X)*W[905] + node1433(X)*W[370] + node1787(X)*W[1708]
    return(result)

@tf.function
def node1259(X):
    result = B[1259] + node1323(X)*W[142]
    return(result)

@tf.function
def node1260(X):
    result = B[1260] + tf.gather(X, 315, axis=1)*W[455] + tf.gather(X, 779, axis=1)*W[1958]
    return(result)

@tf.function
def node1261(X):
    result = B[1261] + tf.gather(X, 627, axis=1)*W[229] + tf.gather(X, 696, axis=1)*W[1113] + tf.gather(X, 732, axis=1)*W[1559] + node1377(X)*W[193] + node1606(X)*W[1944]
    return(result)

@tf.function
def node1262(X):
    result = B[1262] + node1483(X)*W[908] + node1701(X)*W[1526]
    return(result)

@tf.function
def node1263(X):
    result = B[1263]
    return(result)

@tf.function
def node1264(X):
    result = B[1264] + tf.gather(X, 186, axis=1)*W[1557] + tf.gather(X, 256, axis=1)*W[690] + node843(X)*W[879] + node1786(X)*W[109]
    return(result)

@tf.function
def node1265(X):
    result = B[1265] + tf.gather(X, 413, axis=1)*W[1549] + tf.gather(X, 479, axis=1)*W[157] + node1537(X)*W[1927]
    return(result)

@tf.function
def node1266(X):
    result = B[1266] + tf.gather(X, 238, axis=1)*W[887]
    return(result)

@tf.function
def node1267(X):
    result = B[1267]
    return(result)

@tf.function
def node1268(X):
    result = B[1268] + tf.gather(X, 32, axis=1)*W[468] + tf.gather(X, 381, axis=1)*W[1488] + tf.gather(X, 710, axis=1)*W[956] + node839(X)*W[1230]
    return(result)

@tf.function
def node1269(X):
    result = B[1269] + tf.gather(X, 699, axis=1)*W[749]
    return(result)

@tf.function
def node1270(X):
    result = B[1270] + node1144(X)*W[1006] + node1194(X)*W[1930]
    return(result)

@tf.function
def node1271(X):
    result = B[1271] + tf.gather(X, 277, axis=1)*W[534] + node1785(X)*W[1064]
    return(result)

@tf.function
def node1272(X):
    result = B[1272] + node928(X)*W[1617] + node1619(X)*W[1672]
    return(result)

@tf.function
def node1273(X):
    result = B[1273] + node1548(X)*W[787]
    return(result)

@tf.function
def node1274(X):
    result = B[1274] + tf.gather(X, 43, axis=1)*W[503] + node1302(X)*W[418]
    return(result)

@tf.function
def node1275(X):
    result = B[1275] + tf.gather(X, 623, axis=1)*W[1892] + node1216(X)*W[485] + node1635(X)*W[1556]
    return(result)

@tf.function
def node1276(X):
    result = B[1276] + tf.gather(X, 247, axis=1)*W[1000]
    return(result)

@tf.function
def node1277(X):
    result = B[1277] + tf.gather(X, 526, axis=1)*W[811] + node1608(X)*W[1043] + node1765(X)*W[1392]
    return(result)

@tf.function
def node1278(X):
    result = B[1278] + tf.gather(X, 59, axis=1)*W[1642] + node1732(X)*W[1165] + node1758(X)*W[1047]
    return(result)

@tf.function
def node1279(X):
    result = B[1279]
    return(result)

@tf.function
def node1280(X):
    result = B[1280]
    return(result)

@tf.function
def node1281(X):
    result = B[1281] + tf.gather(X, 261, axis=1)*W[893]
    return(result)

@tf.function
def node1282(X):
    result = B[1282] + tf.gather(X, 81, axis=1)*W[1770] + node1772(X)*W[29]
    return(result)

@tf.function
def node1283(X):
    result = B[1283] + tf.gather(X, 115, axis=1)*W[83] + node1779(X)*W[464]
    return(result)

@tf.function
def node1284(X):
    result = B[1284] + tf.gather(X, 231, axis=1)*W[398] + tf.gather(X, 751, axis=1)*W[614]
    return(result)

@tf.function
def node1285(X):
    result = B[1285] + tf.gather(X, 635, axis=1)*W[1485]
    return(result)

@tf.function
def node1286(X):
    result = B[1286]
    return(result)

@tf.function
def node1287(X):
    result = B[1287] + tf.gather(X, 36, axis=1)*W[1626]
    return(result)

@tf.function
def node1288(X):
    result = B[1288] + tf.gather(X, 195, axis=1)*W[1785] + tf.gather(X, 518, axis=1)*W[1197] + node1302(X)*W[1123] + node1706(X)*W[283]
    return(result)

@tf.function
def node1289(X):
    result = B[1289] + tf.gather(X, 283, axis=1)*W[1494] + tf.gather(X, 327, axis=1)*W[1750]
    return(result)

@tf.function
def node1290(X):
    result = B[1290] + tf.gather(X, 57, axis=1)*W[921] + node1154(X)*W[88] + node1390(X)*W[1323] + node1610(X)*W[74]
    return(result)

@tf.function
def node1291(X):
    result = B[1291] + tf.gather(X, 241, axis=1)*W[543] + node1588(X)*W[135]
    return(result)

@tf.function
def node1292(X):
    result = B[1292] + tf.gather(X, 279, axis=1)*W[670]
    return(result)

@tf.function
def node1293(X):
    result = B[1293]
    return(result)

@tf.function
def node1294(X):
    result = B[1294]
    return(result)

@tf.function
def node1295(X):
    result = B[1295] + tf.gather(X, 69, axis=1)*W[746] + tf.gather(X, 258, axis=1)*W[425] + tf.gather(X, 300, axis=1)*W[1180]
    return(result)

@tf.function
def node1296(X):
    result = B[1296] + node900(X)*W[671]
    return(result)

@tf.function
def node1297(X):
    result = B[1297] + tf.gather(X, 397, axis=1)*W[510] + node1774(X)*W[1089]
    return(result)

@tf.function
def node1298(X):
    result = B[1298] + tf.gather(X, 202, axis=1)*W[1514]
    return(result)

@tf.function
def node1299(X):
    result = B[1299] + node1424(X)*W[467]
    return(result)

@tf.function
def node1300(X):
    result = B[1300] + node1782(X)*W[1213]
    return(result)

@tf.function
def node1301(X):
    result = B[1301] + tf.gather(X, 781, axis=1)*W[367] + node1008(X)*W[1994]
    return(result)

@tf.function
def node1302(X):
    result = B[1302] + tf.gather(X, 612, axis=1)*W[704] + node1043(X)*W[914] + node1102(X)*W[1819] + node1155(X)*W[1285]
    return(result)

@tf.function
def node1303(X):
    result = B[1303]
    return(result)

@tf.function
def node1304(X):
    result = B[1304] + tf.gather(X, 96, axis=1)*W[565] + tf.gather(X, 625, axis=1)*W[1035] + node1031(X)*W[163] + node1700(X)*W[96]
    return(result)

@tf.function
def node1305(X):
    result = B[1305] + node941(X)*W[419] + node1757(X)*W[1227]
    return(result)

@tf.function
def node1306(X):
    result = B[1306] + tf.gather(X, 245, axis=1)*W[1308] + node1676(X)*W[1143]
    return(result)

@tf.function
def node1307(X):
    result = B[1307] + tf.gather(X, 683, axis=1)*W[1503]
    return(result)

@tf.function
def node1308(X):
    result = B[1308] + node961(X)*W[1013]
    return(result)

@tf.function
def node1309(X):
    result = B[1309] + tf.gather(X, 672, axis=1)*W[1144]
    return(result)

@tf.function
def node1310(X):
    result = B[1310]
    return(result)

@tf.function
def node1311(X):
    result = B[1311] + node1330(X)*W[1319] + node1668(X)*W[1896]
    return(result)

@tf.function
def node1312(X):
    result = B[1312] + tf.gather(X, 105, axis=1)*W[661]
    return(result)

@tf.function
def node1313(X):
    result = B[1313] + tf.gather(X, 556, axis=1)*W[737] + tf.gather(X, 742, axis=1)*W[601]
    return(result)

@tf.function
def node1314(X):
    result = B[1314] + tf.gather(X, 453, axis=1)*W[182] + node1434(X)*W[740] + node1691(X)*W[677] + node1745(X)*W[452]
    return(result)

@tf.function
def node1315(X):
    result = B[1315] + node1524(X)*W[1001] + node1654(X)*W[863] + node1675(X)*W[587]
    return(result)

@tf.function
def node1316(X):
    result = B[1316] + node899(X)*W[1209] + node1072(X)*W[1792] + node1437(X)*W[1475] + node1555(X)*W[871]
    return(result)

@tf.function
def node1317(X):
    result = B[1317]
    return(result)

@tf.function
def node1318(X):
    result = B[1318] + node853(X)*W[1829] + node1431(X)*W[1052]
    return(result)

@tf.function
def node1319(X):
    result = B[1319] + node872(X)*W[1982] + node1442(X)*W[1992]
    return(result)

@tf.function
def node1320(X):
    result = B[1320] + node1685(X)*W[962]
    return(result)

@tf.function
def node1321(X):
    result = B[1321]
    return(result)

@tf.function
def node1322(X):
    result = B[1322] + node900(X)*W[1604]
    return(result)

@tf.function
def node1323(X):
    result = B[1323] + node1713(X)*W[1347]
    return(result)

@tf.function
def node1324(X):
    result = B[1324] + tf.gather(X, 295, axis=1)*W[469] + tf.gather(X, 465, axis=1)*W[1434]
    return(result)

@tf.function
def node1325(X):
    result = B[1325] + node1049(X)*W[280] + node1135(X)*W[1962]
    return(result)

@tf.function
def node1326(X):
    result = B[1326] + tf.gather(X, 576, axis=1)*W[1856]
    return(result)

@tf.function
def node1327(X):
    result = B[1327] + tf.gather(X, 184, axis=1)*W[825] + node1758(X)*W[1681]
    return(result)

@tf.function
def node1328(X):
    result = B[1328]
    return(result)

@tf.function
def node1329(X):
    result = B[1329] + tf.gather(X, 403, axis=1)*W[1438] + node1020(X)*W[569]
    return(result)

@tf.function
def node1330(X):
    result = B[1330] + node945(X)*W[1860] + node1153(X)*W[390]
    return(result)

@tf.function
def node1331(X):
    result = B[1331] + tf.gather(X, 504, axis=1)*W[104] + node830(X)*W[617] + node1750(X)*W[1641]
    return(result)

@tf.function
def node1332(X):
    result = B[1332] + node1639(X)*W[221] + node1773(X)*W[352]
    return(result)

@tf.function
def node1333(X):
    result = B[1333] + node1073(X)*W[1076]
    return(result)

@tf.function
def node1334(X):
    result = B[1334] + tf.gather(X, 609, axis=1)*W[924] + node1143(X)*W[402] + node1341(X)*W[1541]
    return(result)

@tf.function
def node1335(X):
    result = B[1335] + node1096(X)*W[1218] + node1229(X)*W[1759] + node1675(X)*W[101]
    return(result)

@tf.function
def node1336(X):
    result = B[1336] + tf.gather(X, 50, axis=1)*W[1577] + tf.gather(X, 188, axis=1)*W[1452] + tf.gather(X, 776, axis=1)*W[189]
    return(result)

@tf.function
def node1337(X):
    result = B[1337] + node987(X)*W[1609] + node1001(X)*W[11] + node1488(X)*W[1789]
    return(result)

@tf.function
def node1338(X):
    result = B[1338] + node830(X)*W[822] + node1602(X)*W[1920] + node1698(X)*W[1082] + node1715(X)*W[1023]
    return(result)

@tf.function
def node1339(X):
    result = B[1339] + tf.gather(X, 369, axis=1)*W[1744] + tf.gather(X, 588, axis=1)*W[1132] + tf.gather(X, 615, axis=1)*W[1655] + node901(X)*W[1679] + node1181(X)*W[1316] + node1306(X)*W[1647] + node1395(X)*W[890]
    return(result)

@tf.function
def node1340(X):
    result = B[1340] + node818(X)*W[701]
    return(result)

@tf.function
def node1341(X):
    result = B[1341]
    return(result)

@tf.function
def node1342(X):
    result = B[1342] + tf.gather(X, 422, axis=1)*W[273] + tf.gather(X, 572, axis=1)*W[1398] + node858(X)*W[72]
    return(result)

@tf.function
def node1343(X):
    result = B[1343] + node957(X)*W[655] + node1505(X)*W[1458]
    return(result)

@tf.function
def node1344(X):
    result = B[1344]
    return(result)

@tf.function
def node1345(X):
    result = B[1345] + tf.gather(X, 56, axis=1)*W[1396] + tf.gather(X, 313, axis=1)*W[841] + tf.gather(X, 701, axis=1)*W[198]
    return(result)

@tf.function
def node1346(X):
    result = B[1346] + tf.gather(X, 311, axis=1)*W[130] + node1293(X)*W[81]
    return(result)

@tf.function
def node1347(X):
    result = B[1347] + node802(X)*W[1610] + node836(X)*W[1690]
    return(result)

@tf.function
def node1348(X):
    result = B[1348] + tf.gather(X, 584, axis=1)*W[205] + node900(X)*W[1931] + node1546(X)*W[1776]
    return(result)

@tf.function
def node1349(X):
    result = B[1349]
    return(result)

@tf.function
def node1350(X):
    result = B[1350] + tf.gather(X, 134, axis=1)*W[118]
    return(result)

@tf.function
def node1351(X):
    result = B[1351] + node867(X)*W[463]
    return(result)

@tf.function
def node1352(X):
    result = B[1352] + tf.gather(X, 101, axis=1)*W[365] + tf.gather(X, 213, axis=1)*W[477] + tf.gather(X, 424, axis=1)*W[691] + tf.gather(X, 584, axis=1)*W[1151] + node1500(X)*W[1420] + node1590(X)*W[857]
    return(result)

@tf.function
def node1353(X):
    result = B[1353] + tf.gather(X, 118, axis=1)*W[107] + tf.gather(X, 285, axis=1)*W[1445] + node943(X)*W[1603] + node1754(X)*W[1263]
    return(result)

@tf.function
def node1354(X):
    result = B[1354] + node965(X)*W[1262] + node1142(X)*W[1095]
    return(result)

@tf.function
def node1355(X):
    result = B[1355] + tf.gather(X, 143, axis=1)*W[517] + tf.gather(X, 628, axis=1)*W[943] + node1313(X)*W[1973] + node1401(X)*W[1524]
    return(result)

@tf.function
def node1356(X):
    result = B[1356] + node989(X)*W[1416] + node1053(X)*W[1972]
    return(result)

@tf.function
def node1357(X):
    result = B[1357] + tf.gather(X, 219, axis=1)*W[1234] + tf.gather(X, 469, axis=1)*W[1269] + node1395(X)*W[1100]
    return(result)

@tf.function
def node1358(X):
    result = B[1358] + node1664(X)*W[51]
    return(result)

@tf.function
def node1359(X):
    result = B[1359]
    return(result)

@tf.function
def node1360(X):
    result = B[1360] + tf.gather(X, 122, axis=1)*W[1879] + node1356(X)*W[829]
    return(result)

@tf.function
def node1361(X):
    result = B[1361] + tf.gather(X, 627, axis=1)*W[1273] + tf.gather(X, 700, axis=1)*W[1109] + node1100(X)*W[840] + node1192(X)*W[1479] + node1620(X)*W[1542]
    return(result)

@tf.function
def node1362(X):
    result = B[1362] + tf.gather(X, 647, axis=1)*W[1450] + tf.gather(X, 733, axis=1)*W[546]
    return(result)

@tf.function
def node1363(X):
    result = B[1363] + tf.gather(X, 630, axis=1)*W[173]
    return(result)

@tf.function
def node1364(X):
    result = B[1364] + tf.gather(X, 80, axis=1)*W[1129] + tf.gather(X, 344, axis=1)*W[654] + node1713(X)*W[1211]
    return(result)

@tf.function
def node1365(X):
    result = B[1365]
    return(result)

@tf.function
def node1366(X):
    result = B[1366] + tf.gather(X, 92, axis=1)*W[1478]
    return(result)

@tf.function
def node1367(X):
    result = B[1367]
    return(result)

@tf.function
def node1368(X):
    result = B[1368] + node1196(X)*W[190]
    return(result)

@tf.function
def node1369(X):
    result = B[1369] + tf.gather(X, 145, axis=1)*W[1998] + node1076(X)*W[572]
    return(result)

@tf.function
def node1370(X):
    result = B[1370] + tf.gather(X, 664, axis=1)*W[89]
    return(result)

@tf.function
def node1371(X):
    result = B[1371] + tf.gather(X, 40, axis=1)*W[382] + tf.gather(X, 716, axis=1)*W[1737] + node1624(X)*W[1805]
    return(result)

@tf.function
def node1372(X):
    result = B[1372]
    return(result)

@tf.function
def node1373(X):
    result = B[1373] + tf.gather(X, 369, axis=1)*W[1870] + node1263(X)*W[1466] + node1562(X)*W[1397]
    return(result)

@tf.function
def node1374(X):
    result = B[1374] + node1361(X)*W[1871]
    return(result)

@tf.function
def node1375(X):
    result = B[1375] + tf.gather(X, 621, axis=1)*W[1777]
    return(result)

@tf.function
def node1376(X):
    result = B[1376] + node1382(X)*W[1676]
    return(result)

@tf.function
def node1377(X):
    result = B[1377] + node813(X)*W[520] + node926(X)*W[1164]
    return(result)

@tf.function
def node1378(X):
    result = B[1378]
    return(result)

@tf.function
def node1379(X):
    result = B[1379] + tf.gather(X, 376, axis=1)*W[1740] + tf.gather(X, 576, axis=1)*W[180] + tf.gather(X, 648, axis=1)*W[479] + node840(X)*W[803] + node982(X)*W[249] + node1127(X)*W[1423] + node1331(X)*W[557]
    return(result)

@tf.function
def node1380(X):
    result = B[1380] + tf.gather(X, 131, axis=1)*W[1713] + node1047(X)*W[1257] + node1175(X)*W[1752] + node1513(X)*W[756] + node1536(X)*W[447] + node1610(X)*W[634]
    return(result)

@tf.function
def node1381(X):
    result = B[1381] + tf.gather(X, 393, axis=1)*W[1667] + tf.gather(X, 423, axis=1)*W[1932] + tf.gather(X, 754, axis=1)*W[1066] + node1687(X)*W[1629]
    return(result)

@tf.function
def node1382(X):
    result = B[1382] + node1183(X)*W[545] + node1309(X)*W[1657]
    return(result)

@tf.function
def node1383(X):
    result = B[1383] + tf.gather(X, 314, axis=1)*W[16] + tf.gather(X, 541, axis=1)*W[170]
    return(result)

@tf.function
def node1384(X):
    result = B[1384] + node1760(X)*W[292]
    return(result)

@tf.function
def node1385(X):
    result = B[1385] + node1349(X)*W[48] + node1594(X)*W[1037]
    return(result)

@tf.function
def node1386(X):
    result = B[1386] + tf.gather(X, 631, axis=1)*W[156]
    return(result)

@tf.function
def node1387(X):
    result = B[1387] + node833(X)*W[1505] + node904(X)*W[460] + node934(X)*W[1028]
    return(result)

@tf.function
def node1388(X):
    result = B[1388]
    return(result)

@tf.function
def node1389(X):
    result = B[1389] + tf.gather(X, 42, axis=1)*W[354] + tf.gather(X, 609, axis=1)*W[1509] + node1121(X)*W[1535] + node1466(X)*W[1231]
    return(result)

@tf.function
def node1390(X):
    result = B[1390] + tf.gather(X, 648, axis=1)*W[795] + tf.gather(X, 657, axis=1)*W[1546]
    return(result)

@tf.function
def node1391(X):
    result = B[1391] + tf.gather(X, 106, axis=1)*W[1469]
    return(result)

@tf.function
def node1392(X):
    result = B[1392] + tf.gather(X, 327, axis=1)*W[1272] + tf.gather(X, 618, axis=1)*W[649] + tf.gather(X, 770, axis=1)*W[1356] + node899(X)*W[781]
    return(result)

@tf.function
def node1393(X):
    result = B[1393] + tf.gather(X, 599, axis=1)*W[602] + node1694(X)*W[1499]
    return(result)

@tf.function
def node1394(X):
    result = B[1394] + tf.gather(X, 302, axis=1)*W[1778]
    return(result)

@tf.function
def node1395(X):
    result = B[1395]
    return(result)

@tf.function
def node1396(X):
    result = B[1396]
    return(result)

@tf.function
def node1397(X):
    result = B[1397] + tf.gather(X, 80, axis=1)*W[58] + tf.gather(X, 584, axis=1)*W[676] + tf.gather(X, 647, axis=1)*W[1220] + tf.gather(X, 757, axis=1)*W[208]
    return(result)

@tf.function
def node1398(X):
    result = B[1398]
    return(result)

@tf.function
def node1399(X):
    result = B[1399] + node1068(X)*W[776]
    return(result)

@tf.function
def node1400(X):
    result = B[1400]
    return(result)

@tf.function
def node1401(X):
    result = B[1401] + node1648(X)*W[774]
    return(result)

@tf.function
def node1402(X):
    result = B[1402] + node810(X)*W[1244]
    return(result)

@tf.function
def node1403(X):
    result = B[1403] + node988(X)*W[659] + node1080(X)*W[1659] + node1170(X)*W[872]
    return(result)

@tf.function
def node1404(X):
    result = B[1404] + tf.gather(X, 314, axis=1)*W[366] + tf.gather(X, 495, axis=1)*W[482]
    return(result)

@tf.function
def node1405(X):
    result = B[1405] + tf.gather(X, 408, axis=1)*W[1997] + tf.gather(X, 699, axis=1)*W[197] + tf.gather(X, 783, axis=1)*W[1353] + node1219(X)*W[490]
    return(result)

@tf.function
def node1406(X):
    result = B[1406]
    return(result)

@tf.function
def node1407(X):
    result = B[1407] + tf.gather(X, 360, axis=1)*W[1016] + node1088(X)*W[1866] + node1379(X)*W[97] + node1641(X)*W[297]
    return(result)

@tf.function
def node1408(X):
    result = B[1408] + tf.gather(X, 16, axis=1)*W[1114] + node950(X)*W[733]
    return(result)

@tf.function
def node1409(X):
    result = B[1409] + node1361(X)*W[982]
    return(result)

@tf.function
def node1410(X):
    result = B[1410] + tf.gather(X, 108, axis=1)*W[206] + tf.gather(X, 407, axis=1)*W[1259] + node1252(X)*W[837]
    return(result)

@tf.function
def node1411(X):
    result = B[1411] + tf.gather(X, 22, axis=1)*W[179] + node933(X)*W[1756]
    return(result)

@tf.function
def node1412(X):
    result = B[1412] + tf.gather(X, 325, axis=1)*W[1818] + node1776(X)*W[1824]
    return(result)

@tf.function
def node1413(X):
    result = B[1413] + tf.gather(X, 7, axis=1)*W[933]
    return(result)

@tf.function
def node1414(X):
    result = B[1414] + tf.gather(X, 37, axis=1)*W[55] + tf.gather(X, 665, axis=1)*W[379] + node1099(X)*W[263]
    return(result)

@tf.function
def node1415(X):
    result = B[1415] + tf.gather(X, 63, axis=1)*W[939] + node1478(X)*W[134]
    return(result)

@tf.function
def node1416(X):
    result = B[1416]
    return(result)

@tf.function
def node1417(X):
    result = B[1417] + tf.gather(X, 633, axis=1)*W[950] + node1508(X)*W[1483]
    return(result)

@tf.function
def node1418(X):
    result = B[1418] + node1081(X)*W[1884]
    return(result)

@tf.function
def node1419(X):
    result = B[1419] + tf.gather(X, 416, axis=1)*W[1790] + tf.gather(X, 504, axis=1)*W[1400] + tf.gather(X, 737, axis=1)*W[311]
    return(result)

@tf.function
def node1420(X):
    result = B[1420]
    return(result)

@tf.function
def node1421(X):
    result = B[1421] + tf.gather(X, 374, axis=1)*W[828] + tf.gather(X, 623, axis=1)*W[810] + tf.gather(X, 669, axis=1)*W[1652] + node890(X)*W[680]
    return(result)

@tf.function
def node1422(X):
    result = B[1422] + tf.gather(X, 45, axis=1)*W[1686] + tf.gather(X, 634, axis=1)*W[1075]
    return(result)

@tf.function
def node1423(X):
    result = B[1423] + tf.gather(X, 491, axis=1)*W[911] + node847(X)*W[1160]
    return(result)

@tf.function
def node1424(X):
    result = B[1424] + tf.gather(X, 64, axis=1)*W[608]
    return(result)

@tf.function
def node1425(X):
    result = B[1425] + tf.gather(X, 297, axis=1)*W[607]
    return(result)

@tf.function
def node1426(X):
    result = B[1426] + tf.gather(X, 108, axis=1)*W[622] + node1096(X)*W[500] + node1701(X)*W[1865]
    return(result)

@tf.function
def node1427(X):
    result = B[1427] + tf.gather(X, 488, axis=1)*W[305] + tf.gather(X, 492, axis=1)*W[1852]
    return(result)

@tf.function
def node1428(X):
    result = B[1428] + node1761(X)*W[613]
    return(result)

@tf.function
def node1429(X):
    result = B[1429] + tf.gather(X, 263, axis=1)*W[988] + node911(X)*W[1587] + node927(X)*W[1795] + node1547(X)*W[1282] + node1708(X)*W[846]
    return(result)

@tf.function
def node1430(X):
    result = B[1430] + node1282(X)*W[1414]
    return(result)

@tf.function
def node1431(X):
    result = B[1431] + tf.gather(X, 604, axis=1)*W[138] + node1610(X)*W[1415] + node1788(X)*W[800]
    return(result)

@tf.function
def node1432(X):
    result = B[1432] + tf.gather(X, 212, axis=1)*W[1199] + tf.gather(X, 476, axis=1)*W[1627] + node1675(X)*W[1495]
    return(result)

@tf.function
def node1433(X):
    result = B[1433] + tf.gather(X, 493, axis=1)*W[1174]
    return(result)

@tf.function
def node1434(X):
    result = B[1434] + node803(X)*W[1512]
    return(result)

@tf.function
def node1435(X):
    result = B[1435]
    return(result)

@tf.function
def node1436(X):
    result = B[1436] + tf.gather(X, 557, axis=1)*W[549] + node1313(X)*W[626] + node1487(X)*W[801]
    return(result)

@tf.function
def node1437(X):
    result = B[1437] + tf.gather(X, 750, axis=1)*W[1798]
    return(result)

@tf.function
def node1438(X):
    result = B[1438] + node1240(X)*W[966] + node1782(X)*W[125]
    return(result)

@tf.function
def node1439(X):
    result = B[1439] + node1180(X)*W[897]
    return(result)

@tf.function
def node1440(X):
    result = B[1440] + tf.gather(X, 221, axis=1)*W[1021]
    return(result)

@tf.function
def node1441(X):
    result = B[1441] + tf.gather(X, 68, axis=1)*W[308] + node826(X)*W[732] + node1550(X)*W[1500]
    return(result)

@tf.function
def node1442(X):
    result = B[1442] + tf.gather(X, 164, axis=1)*W[1010] + node1491(X)*W[1763] + node1549(X)*W[1801] + node1659(X)*W[395]
    return(result)

@tf.function
def node1443(X):
    result = B[1443] + tf.gather(X, 106, axis=1)*W[619]
    return(result)

@tf.function
def node1444(X):
    result = B[1444] + tf.gather(X, 518, axis=1)*W[742]
    return(result)

@tf.function
def node1445(X):
    result = B[1445] + node1224(X)*W[1799] + node1722(X)*W[1250]
    return(result)

@tf.function
def node1446(X):
    result = B[1446] + tf.gather(X, 205, axis=1)*W[799]
    return(result)

@tf.function
def node1447(X):
    result = B[1447] + tf.gather(X, 483, axis=1)*W[1692] + node1084(X)*W[1980] + node1552(X)*W[470] + node1739(X)*W[326]
    return(result)

@tf.function
def node1448(X):
    result = B[1448] + tf.gather(X, 572, axis=1)*W[963] + node1663(X)*W[1031]
    return(result)

@tf.function
def node1449(X):
    result = B[1449] + tf.gather(X, 165, axis=1)*W[288] + tf.gather(X, 375, axis=1)*W[1888] + node1764(X)*W[648]
    return(result)

@tf.function
def node1450(X):
    result = B[1450] + node1084(X)*W[1058] + node1433(X)*W[687]
    return(result)

@tf.function
def node1451(X):
    result = B[1451] + tf.gather(X, 288, axis=1)*W[918] + tf.gather(X, 493, axis=1)*W[151] + node828(X)*W[260]
    return(result)

@tf.function
def node1452(X):
    result = B[1452] + tf.gather(X, 204, axis=1)*W[1320] + node1273(X)*W[172]
    return(result)

@tf.function
def node1453(X):
    result = B[1453] + node1634(X)*W[254] + node1700(X)*W[110]
    return(result)

@tf.function
def node1454(X):
    result = B[1454] + tf.gather(X, 685, axis=1)*W[115] + node1409(X)*W[1413] + node1787(X)*W[506]
    return(result)

@tf.function
def node1455(X):
    result = B[1455] + tf.gather(X, 128, axis=1)*W[1660] + node1214(X)*W[1585]
    return(result)

@tf.function
def node1456(X):
    result = B[1456] + node954(X)*W[60]
    return(result)

@tf.function
def node1457(X):
    result = B[1457] + node1328(X)*W[1405]
    return(result)

@tf.function
def node1458(X):
    result = B[1458] + node944(X)*W[1554]
    return(result)

@tf.function
def node1459(X):
    result = B[1459]
    return(result)

@tf.function
def node1460(X):
    result = B[1460] + node891(X)*W[984] + node1553(X)*W[1900] + node1775(X)*W[435]
    return(result)

@tf.function
def node1461(X):
    result = B[1461] + tf.gather(X, 550, axis=1)*W[1562] + node1485(X)*W[466]
    return(result)

@tf.function
def node1462(X):
    result = B[1462] + tf.gather(X, 375, axis=1)*W[888] + node1420(X)*W[1551] + node1552(X)*W[656]
    return(result)

@tf.function
def node1463(X):
    result = B[1463] + node1130(X)*W[1206]
    return(result)

@tf.function
def node1464(X):
    result = B[1464] + tf.gather(X, 19, axis=1)*W[1239]
    return(result)

@tf.function
def node1465(X):
    result = B[1465] + node1252(X)*W[1531]
    return(result)

@tf.function
def node1466(X):
    result = B[1466] + tf.gather(X, 748, axis=1)*W[1675] + node808(X)*W[224] + node1017(X)*W[44]
    return(result)

@tf.function
def node1467(X):
    result = B[1467] + tf.gather(X, 0, axis=1)*W[1212] + tf.gather(X, 202, axis=1)*W[548] + tf.gather(X, 210, axis=1)*W[246] + tf.gather(X, 300, axis=1)*W[1697] + tf.gather(X, 463, axis=1)*W[204] + node847(X)*W[535] + node1013(X)*W[76]
    return(result)

@tf.function
def node1468(X):
    result = B[1468] + tf.gather(X, 413, axis=1)*W[1268] + tf.gather(X, 758, axis=1)*W[851] + node1451(X)*W[1718]
    return(result)

@tf.function
def node1469(X):
    result = B[1469]
    return(result)

@tf.function
def node1470(X):
    result = B[1470] + tf.gather(X, 100, axis=1)*W[672] + tf.gather(X, 306, axis=1)*W[969] + tf.gather(X, 358, axis=1)*W[1429] + tf.gather(X, 730, axis=1)*W[1964]
    return(result)

@tf.function
def node1471(X):
    result = B[1471] + tf.gather(X, 392, axis=1)*W[1730] + tf.gather(X, 718, axis=1)*W[1705]
    return(result)

@tf.function
def node1472(X):
    result = B[1472] + node1500(X)*W[1841]
    return(result)

@tf.function
def node1473(X):
    result = B[1473]
    return(result)

@tf.function
def node1474(X):
    result = B[1474] + tf.gather(X, 272, axis=1)*W[282] + tf.gather(X, 398, axis=1)*W[213]
    return(result)

@tf.function
def node1475(X):
    result = B[1475] + node1298(X)*W[369]
    return(result)

@tf.function
def node1476(X):
    result = B[1476] + tf.gather(X, 268, axis=1)*W[1877] + tf.gather(X, 322, axis=1)*W[499] + tf.gather(X, 725, axis=1)*W[1545] + node1432(X)*W[1249]
    return(result)

@tf.function
def node1477(X):
    result = B[1477] + tf.gather(X, 285, axis=1)*W[1711] + tf.gather(X, 646, axis=1)*W[1914] + tf.gather(X, 703, axis=1)*W[709] + node1363(X)*W[1706] + node1755(X)*W[1804]
    return(result)

@tf.function
def node1478(X):
    result = B[1478] + tf.gather(X, 349, axis=1)*W[1394] + node1017(X)*W[971]
    return(result)

@tf.function
def node1479(X):
    result = B[1479] + node847(X)*W[248] + node897(X)*W[1079]
    return(result)

@tf.function
def node1480(X):
    result = B[1480] + node937(X)*W[353] + node1105(X)*W[1176]
    return(result)

@tf.function
def node1481(X):
    result = B[1481] + node976(X)*W[833]
    return(result)

@tf.function
def node1482(X):
    result = B[1482] + tf.gather(X, 556, axis=1)*W[1446] + node879(X)*W[1987]
    return(result)

@tf.function
def node1483(X):
    result = B[1483] + tf.gather(X, 490, axis=1)*W[1122]
    return(result)

@tf.function
def node1484(X):
    result = B[1484] + node1657(X)*W[754]
    return(result)

@tf.function
def node1485(X):
    result = B[1485] + tf.gather(X, 87, axis=1)*W[1393] + tf.gather(X, 434, axis=1)*W[1355] + node1237(X)*W[66]
    return(result)

@tf.function
def node1486(X):
    result = B[1486] + tf.gather(X, 292, axis=1)*W[155] + tf.gather(X, 342, axis=1)*W[1506] + tf.gather(X, 487, axis=1)*W[1709] + node927(X)*W[30] + node1037(X)*W[1632] + node1129(X)*W[1661]
    return(result)

@tf.function
def node1487(X):
    result = B[1487] + node1307(X)*W[1960]
    return(result)

@tf.function
def node1488(X):
    result = B[1488] + tf.gather(X, 234, axis=1)*W[1828] + node967(X)*W[1851]
    return(result)

@tf.function
def node1489(X):
    result = B[1489] + node1320(X)*W[1810] + node1497(X)*W[1729]
    return(result)

@tf.function
def node1490(X):
    result = B[1490] + tf.gather(X, 31, axis=1)*W[1376] + tf.gather(X, 126, axis=1)*W[1232]
    return(result)

@tf.function
def node1491(X):
    result = B[1491] + tf.gather(X, 556, axis=1)*W[1969]
    return(result)

@tf.function
def node1492(X):
    result = B[1492] + tf.gather(X, 327, axis=1)*W[1103] + tf.gather(X, 544, axis=1)*W[823] + node1015(X)*W[1716] + node1490(X)*W[562] + node1737(X)*W[1301]
    return(result)

@tf.function
def node1493(X):
    result = B[1493] + tf.gather(X, 380, axis=1)*W[1903] + tf.gather(X, 419, axis=1)*W[1337] + node1138(X)*W[1453] + node1225(X)*W[318]
    return(result)

@tf.function
def node1494(X):
    result = B[1494] + tf.gather(X, 91, axis=1)*W[342] + tf.gather(X, 387, axis=1)*W[663]
    return(result)

@tf.function
def node1495(X):
    result = B[1495] + node863(X)*W[1097] + node946(X)*W[1305] + node1539(X)*W[167] + node1711(X)*W[1311]
    return(result)

@tf.function
def node1496(X):
    result = B[1496] + tf.gather(X, 150, axis=1)*W[153]
    return(result)

@tf.function
def node1497(X):
    result = B[1497] + tf.gather(X, 135, axis=1)*W[1574] + tf.gather(X, 254, axis=1)*W[1050]
    return(result)

@tf.function
def node1498(X):
    result = B[1498] + tf.gather(X, 307, axis=1)*W[1317] + node1086(X)*W[514] + node1749(X)*W[274]
    return(result)

@tf.function
def node1499(X):
    result = B[1499] + tf.gather(X, 89, axis=1)*W[1373] + node1503(X)*W[1427] + node1650(X)*W[441]
    return(result)

@tf.function
def node1500(X):
    result = B[1500] + node1656(X)*W[1297]
    return(result)

@tf.function
def node1501(X):
    result = B[1501] + node1618(X)*W[1216]
    return(result)

@tf.function
def node1502(X):
    result = B[1502] + node1136(X)*W[1773] + node1774(X)*W[1312]
    return(result)

@tf.function
def node1503(X):
    result = B[1503] + tf.gather(X, 217, axis=1)*W[1102] + tf.gather(X, 474, axis=1)*W[1850]
    return(result)

@tf.function
def node1504(X):
    result = B[1504] + node1376(X)*W[665] + node1527(X)*W[64]
    return(result)

@tf.function
def node1505(X):
    result = B[1505] + tf.gather(X, 540, axis=1)*W[1990] + tf.gather(X, 582, axis=1)*W[1894] + tf.gather(X, 640, axis=1)*W[862]
    return(result)

@tf.function
def node1506(X):
    result = B[1506] + node1428(X)*W[1595]
    return(result)

@tf.function
def node1507(X):
    result = B[1507] + node1236(X)*W[836] + node1298(X)*W[1271] + node1619(X)*W[1063]
    return(result)

@tf.function
def node1508(X):
    result = B[1508] + tf.gather(X, 232, axis=1)*W[388] + node1311(X)*W[293] + node1450(X)*W[341] + node1569(X)*W[339]
    return(result)

@tf.function
def node1509(X):
    result = B[1509] + tf.gather(X, 362, axis=1)*W[1362]
    return(result)

@tf.function
def node1510(X):
    result = B[1510] + tf.gather(X, 65, axis=1)*W[1843] + node1053(X)*W[119]
    return(result)

@tf.function
def node1511(X):
    result = B[1511] + tf.gather(X, 266, axis=1)*W[1527]
    return(result)

@tf.function
def node1512(X):
    result = B[1512]
    return(result)

@tf.function
def node1513(X):
    result = B[1513] + tf.gather(X, 159, axis=1)*W[54]
    return(result)

@tf.function
def node1514(X):
    result = B[1514] + tf.gather(X, 521, axis=1)*W[1906]
    return(result)

@tf.function
def node1515(X):
    result = B[1515] + tf.gather(X, 677, axis=1)*W[1329] + node929(X)*W[113] + node1761(X)*W[567]
    return(result)

@tf.function
def node1516(X):
    result = B[1516] + tf.gather(X, 120, axis=1)*W[59]
    return(result)

@tf.function
def node1517(X):
    result = B[1517] + tf.gather(X, 333, axis=1)*W[1291] + node1761(X)*W[612]
    return(result)

@tf.function
def node1518(X):
    result = B[1518] + node1078(X)*W[832]
    return(result)

@tf.function
def node1519(X):
    result = B[1519] + tf.gather(X, 333, axis=1)*W[191] + node1463(X)*W[312]
    return(result)

@tf.function
def node1520(X):
    result = B[1520] + tf.gather(X, 312, axis=1)*W[509] + node1170(X)*W[1949]
    return(result)

@tf.function
def node1521(X):
    result = B[1521] + tf.gather(X, 221, axis=1)*W[1904] + node959(X)*W[790] + node1065(X)*W[1489] + node1450(X)*W[1868] + node1597(X)*W[541]
    return(result)

@tf.function
def node1522(X):
    result = B[1522] + tf.gather(X, 413, axis=1)*W[711] + tf.gather(X, 458, axis=1)*W[716] + tf.gather(X, 701, axis=1)*W[82] + node1763(X)*W[552]
    return(result)

@tf.function
def node1523(X):
    result = B[1523] + tf.gather(X, 445, axis=1)*W[1387] + node1104(X)*W[1536] + node1667(X)*W[667]
    return(result)

@tf.function
def node1524(X):
    result = B[1524] + tf.gather(X, 351, axis=1)*W[24] + node1566(X)*W[1189] + node1645(X)*W[146] + node1719(X)*W[1148]
    return(result)

@tf.function
def node1525(X):
    result = B[1525]
    return(result)

@tf.function
def node1526(X):
    result = B[1526] + node1153(X)*W[139]
    return(result)

@tf.function
def node1527(X):
    result = B[1527] + tf.gather(X, 303, axis=1)*W[1440] + tf.gather(X, 327, axis=1)*W[591] + tf.gather(X, 756, axis=1)*W[98] + node1679(X)*W[761]
    return(result)

@tf.function
def node1528(X):
    result = B[1528] + tf.gather(X, 119, axis=1)*W[1378]
    return(result)

@tf.function
def node1529(X):
    result = B[1529]
    return(result)

@tf.function
def node1530(X):
    result = B[1530] + tf.gather(X, 193, axis=1)*W[764] + node1557(X)*W[410]
    return(result)

@tf.function
def node1531(X):
    result = B[1531]
    return(result)

@tf.function
def node1532(X):
    result = B[1532] + tf.gather(X, 411, axis=1)*W[233] + node1736(X)*W[1432]
    return(result)

@tf.function
def node1533(X):
    result = B[1533] + tf.gather(X, 653, axis=1)*W[494] + node870(X)*W[1670] + node1076(X)*W[739] + node1684(X)*W[657]
    return(result)

@tf.function
def node1534(X):
    result = B[1534] + tf.gather(X, 358, axis=1)*W[1278] + node1117(X)*W[456]
    return(result)

@tf.function
def node1535(X):
    result = B[1535] + tf.gather(X, 327, axis=1)*W[706] + tf.gather(X, 386, axis=1)*W[1096]
    return(result)

@tf.function
def node1536(X):
    result = B[1536] + tf.gather(X, 358, axis=1)*W[633]
    return(result)

@tf.function
def node1537(X):
    result = B[1537] + tf.gather(X, 681, axis=1)*W[1459] + node1261(X)*W[1735]
    return(result)

@tf.function
def node1538(X):
    result = B[1538] + tf.gather(X, 653, axis=1)*W[237] + node1243(X)*W[542] + node1283(X)*W[936] + node1453(X)*W[1565]
    return(result)

@tf.function
def node1539(X):
    result = B[1539] + tf.gather(X, 103, axis=1)*W[1324] + node1396(X)*W[1084]
    return(result)

@tf.function
def node1540(X):
    result = B[1540] + tf.gather(X, 569, axis=1)*W[1511] + node1455(X)*W[735]
    return(result)

@tf.function
def node1541(X):
    result = B[1541]
    return(result)

@tf.function
def node1542(X):
    result = B[1542]
    return(result)

@tf.function
def node1543(X):
    result = B[1543]
    return(result)

@tf.function
def node1544(X):
    result = B[1544] + tf.gather(X, 152, axis=1)*W[1835] + tf.gather(X, 379, axis=1)*W[1055]
    return(result)

@tf.function
def node1545(X):
    result = B[1545] + tf.gather(X, 165, axis=1)*W[1395] + node1608(X)*W[1543]
    return(result)

@tf.function
def node1546(X):
    result = B[1546] + tf.gather(X, 154, axis=1)*W[1710] + tf.gather(X, 266, axis=1)*W[881] + tf.gather(X, 296, axis=1)*W[100] + node1059(X)*W[397]
    return(result)

@tf.function
def node1547(X):
    result = B[1547] + tf.gather(X, 648, axis=1)*W[873]
    return(result)

@tf.function
def node1548(X):
    result = B[1548] + tf.gather(X, 711, axis=1)*W[512] + node1655(X)*W[1890]
    return(result)

@tf.function
def node1549(X):
    result = B[1549]
    return(result)

@tf.function
def node1550(X):
    result = B[1550] + tf.gather(X, 673, axis=1)*W[1723] + node1054(X)*W[343]
    return(result)

@tf.function
def node1551(X):
    result = B[1551] + tf.gather(X, 192, axis=1)*W[1391] + tf.gather(X, 369, axis=1)*W[426] + node1223(X)*W[788] + node1298(X)*W[568] + node1363(X)*W[70]
    return(result)

@tf.function
def node1552(X):
    result = B[1552]
    return(result)

@tf.function
def node1553(X):
    result = B[1553]
    return(result)

@tf.function
def node1554(X):
    result = B[1554] + tf.gather(X, 439, axis=1)*W[1243]
    return(result)

@tf.function
def node1555(X):
    result = B[1555] + node1642(X)*W[188]
    return(result)

@tf.function
def node1556(X):
    result = B[1556] + tf.gather(X, 600, axis=1)*W[1631] + node1472(X)*W[18]
    return(result)

@tf.function
def node1557(X):
    result = B[1557] + tf.gather(X, 65, axis=1)*W[1596] + tf.gather(X, 209, axis=1)*W[502]
    return(result)

@tf.function
def node1558(X):
    result = B[1558] + node1274(X)*W[1855] + node1605(X)*W[692]
    return(result)

@tf.function
def node1559(X):
    result = B[1559] + tf.gather(X, 22, axis=1)*W[362] + node1368(X)*W[92]
    return(result)

@tf.function
def node1560(X):
    result = B[1560] + tf.gather(X, 10, axis=1)*W[161] + node1033(X)*W[604]
    return(result)

@tf.function
def node1561(X):
    result = B[1561] + tf.gather(X, 776, axis=1)*W[1726] + node1496(X)*W[1233]
    return(result)

@tf.function
def node1562(X):
    result = B[1562] + node1227(X)*W[1033] + node1308(X)*W[802]
    return(result)

@tf.function
def node1563(X):
    result = B[1563] + tf.gather(X, 303, axis=1)*W[355] + tf.gather(X, 672, axis=1)*W[815] + tf.gather(X, 705, axis=1)*W[12] + node800(X)*W[1219] + node1270(X)*W[1289]
    return(result)

@tf.function
def node1564(X):
    result = B[1564] + tf.gather(X, 774, axis=1)*W[327]
    return(result)

@tf.function
def node1565(X):
    result = B[1565] + tf.gather(X, 562, axis=1)*W[1029] + tf.gather(X, 564, axis=1)*W[870] + node1343(X)*W[1978] + node1413(X)*W[1288]
    return(result)

@tf.function
def node1566(X):
    result = B[1566] + node1475(X)*W[158]
    return(result)

@tf.function
def node1567(X):
    result = B[1567] + node1149(X)*W[1796] + node1181(X)*W[23] + node1470(X)*W[349]
    return(result)

@tf.function
def node1568(X):
    result = B[1568] + tf.gather(X, 116, axis=1)*W[989] + node1523(X)*W[1836] + node1534(X)*W[1083]
    return(result)

@tf.function
def node1569(X):
    result = B[1569] + tf.gather(X, 777, axis=1)*W[1640]
    return(result)

@tf.function
def node1570(X):
    result = B[1570] + tf.gather(X, 395, axis=1)*W[1111]
    return(result)

@tf.function
def node1571(X):
    result = B[1571] + node1744(X)*W[747]
    return(result)

@tf.function
def node1572(X):
    result = B[1572] + node1019(X)*W[1687] + node1401(X)*W[239]
    return(result)

@tf.function
def node1573(X):
    result = B[1573] + node1184(X)*W[306]
    return(result)

@tf.function
def node1574(X):
    result = B[1574] + tf.gather(X, 179, axis=1)*W[446] + tf.gather(X, 492, axis=1)*W[906] + tf.gather(X, 506, axis=1)*W[1121] + tf.gather(X, 781, axis=1)*W[309] + node1411(X)*W[1721]
    return(result)

@tf.function
def node1575(X):
    result = B[1575] + tf.gather(X, 585, axis=1)*W[770]
    return(result)

@tf.function
def node1576(X):
    result = B[1576] + node1437(X)*W[983]
    return(result)

@tf.function
def node1577(X):
    result = B[1577] + tf.gather(X, 583, axis=1)*W[1142]
    return(result)

@tf.function
def node1578(X):
    result = B[1578] + node1234(X)*W[1712] + node1748(X)*W[436]
    return(result)

@tf.function
def node1579(X):
    result = B[1579] + tf.gather(X, 536, axis=1)*W[493]
    return(result)

@tf.function
def node1580(X):
    result = B[1580]
    return(result)

@tf.function
def node1581(X):
    result = B[1581]
    return(result)

@tf.function
def node1582(X):
    result = B[1582] + tf.gather(X, 92, axis=1)*W[1724] + node1376(X)*W[1266] + node1791(X)*W[1011]
    return(result)

@tf.function
def node1583(X):
    result = B[1583] + node1064(X)*W[1492]
    return(result)

@tf.function
def node1584(X):
    result = B[1584] + tf.gather(X, 136, axis=1)*W[1607] + node941(X)*W[669]
    return(result)

@tf.function
def node1585(X):
    result = B[1585] + tf.gather(X, 735, axis=1)*W[1677] + node1193(X)*W[332]
    return(result)

@tf.function
def node1586(X):
    result = B[1586] + tf.gather(X, 299, axis=1)*W[497] + tf.gather(X, 635, axis=1)*W[128] + node1167(X)*W[203] + node1185(X)*W[1984]
    return(result)

@tf.function
def node1587(X):
    result = B[1587] + tf.gather(X, 39, axis=1)*W[679] + node1124(X)*W[1847] + node1500(X)*W[1380] + node1739(X)*W[1327]
    return(result)

@tf.function
def node1588(X):
    result = B[1588] + tf.gather(X, 304, axis=1)*W[563]
    return(result)

@tf.function
def node1589(X):
    result = B[1589] + tf.gather(X, 218, axis=1)*W[1547]
    return(result)

@tf.function
def node1590(X):
    result = B[1590] + tf.gather(X, 257, axis=1)*W[338] + node1205(X)*W[1433] + node1694(X)*W[1834]
    return(result)

@tf.function
def node1591(X):
    result = B[1591] + tf.gather(X, 442, axis=1)*W[1731] + node1511(X)*W[948]
    return(result)

@tf.function
def node1592(X):
    result = B[1592] + tf.gather(X, 132, axis=1)*W[1191] + tf.gather(X, 551, axis=1)*W[1457] + node1007(X)*W[132] + node1534(X)*W[1077]
    return(result)

@tf.function
def node1593(X):
    result = B[1593] + node980(X)*W[1967]
    return(result)

@tf.function
def node1594(X):
    result = B[1594]
    return(result)

@tf.function
def node1595(X):
    result = B[1595] + tf.gather(X, 431, axis=1)*W[407] + node1763(X)*W[573]
    return(result)

@tf.function
def node1596(X):
    result = B[1596] + node1243(X)*W[145]
    return(result)

@tf.function
def node1597(X):
    result = B[1597]
    return(result)

@tf.function
def node1598(X):
    result = B[1598] + tf.gather(X, 146, axis=1)*W[15]
    return(result)

@tf.function
def node1599(X):
    result = B[1599]
    return(result)

@tf.function
def node1600(X):
    result = B[1600] + tf.gather(X, 538, axis=1)*W[1537]
    return(result)

@tf.function
def node1601(X):
    result = B[1601] + tf.gather(X, 91, axis=1)*W[1310] + tf.gather(X, 156, axis=1)*W[1722] + tf.gather(X, 467, axis=1)*W[1404] + tf.gather(X, 737, axis=1)*W[214] + node1161(X)*W[183] + node1279(X)*W[1448]
    return(result)

@tf.function
def node1602(X):
    result = B[1602] + tf.gather(X, 395, axis=1)*W[738] + node827(X)*W[272]
    return(result)

@tf.function
def node1603(X):
    result = B[1603] + tf.gather(X, 431, axis=1)*W[1490]
    return(result)

@tf.function
def node1604(X):
    result = B[1604] + tf.gather(X, 16, axis=1)*W[1126]
    return(result)

@tf.function
def node1605(X):
    result = B[1605] + tf.gather(X, 75, axis=1)*W[1859] + tf.gather(X, 311, axis=1)*W[736]
    return(result)

@tf.function
def node1606(X):
    result = B[1606] + tf.gather(X, 158, axis=1)*W[1365] + tf.gather(X, 733, axis=1)*W[476]
    return(result)

@tf.function
def node1607(X):
    result = B[1607] + node1384(X)*W[641]
    return(result)

@tf.function
def node1608(X):
    result = B[1608] + node809(X)*W[1463] + node816(X)*W[1188] + node981(X)*W[1118] + node1709(X)*W[1696]
    return(result)

@tf.function
def node1609(X):
    result = B[1609]
    return(result)

@tf.function
def node1610(X):
    result = B[1610] + node1281(X)*W[1606] + node1743(X)*W[1481]
    return(result)

@tf.function
def node1611(X):
    result = B[1611]
    return(result)

@tf.function
def node1612(X):
    result = B[1612] + tf.gather(X, 100, axis=1)*W[807] + node1000(X)*W[1704] + node1650(X)*W[794]
    return(result)

@tf.function
def node1613(X):
    result = B[1613] + node1741(X)*W[278]
    return(result)

@tf.function
def node1614(X):
    result = B[1614] + node915(X)*W[1071] + node1175(X)*W[731]
    return(result)

@tf.function
def node1615(X):
    result = B[1615]
    return(result)

@tf.function
def node1616(X):
    result = B[1616] + tf.gather(X, 95, axis=1)*W[913] + node1053(X)*W[137] + node1555(X)*W[594] + node1596(X)*W[556]
    return(result)

@tf.function
def node1617(X):
    result = B[1617] + tf.gather(X, 375, axis=1)*W[358] + node1240(X)*W[1296]
    return(result)

@tf.function
def node1618(X):
    result = B[1618] + node1580(X)*W[1754]
    return(result)

@tf.function
def node1619(X):
    result = B[1619] + node1474(X)*W[94]
    return(result)

@tf.function
def node1620(X):
    result = B[1620] + tf.gather(X, 177, axis=1)*W[1883] + tf.gather(X, 703, axis=1)*W[262]
    return(result)

@tf.function
def node1621(X):
    result = B[1621] + tf.gather(X, 220, axis=1)*W[1946] + node1149(X)*W[498] + node1575(X)*W[728]
    return(result)

@tf.function
def node1622(X):
    result = B[1622] + tf.gather(X, 643, axis=1)*W[990] + node1411(X)*W[1878]
    return(result)

@tf.function
def node1623(X):
    result = B[1623] + tf.gather(X, 37, axis=1)*W[1039] + tf.gather(X, 314, axis=1)*W[112] + tf.gather(X, 711, axis=1)*W[480] + tf.gather(X, 764, axis=1)*W[317] + node913(X)*W[50] + node1631(X)*W[491] + node1649(X)*W[1875]
    return(result)

@tf.function
def node1624(X):
    result = B[1624] + tf.gather(X, 619, axis=1)*W[1862] + node1116(X)*W[606]
    return(result)

@tf.function
def node1625(X):
    result = B[1625] + node1314(X)*W[393]
    return(result)

@tf.function
def node1626(X):
    result = B[1626] + tf.gather(X, 34, axis=1)*W[1593] + node950(X)*W[1153]
    return(result)

@tf.function
def node1627(X):
    result = B[1627] + tf.gather(X, 4, axis=1)*W[643] + node1243(X)*W[1793]
    return(result)

@tf.function
def node1628(X):
    result = B[1628] + tf.gather(X, 4, axis=1)*W[359] + tf.gather(X, 605, axis=1)*W[1797] + node1241(X)*W[1134]
    return(result)

@tf.function
def node1629(X):
    result = B[1629] + node805(X)*W[445]
    return(result)

@tf.function
def node1630(X):
    result = B[1630] + tf.gather(X, 331, axis=1)*W[1751] + tf.gather(X, 539, axis=1)*W[32] + tf.gather(X, 615, axis=1)*W[1484] + node1124(X)*W[1240] + node1207(X)*W[1637] + node1728(X)*W[994] + node1740(X)*W[958]
    return(result)

@tf.function
def node1631(X):
    result = B[1631] + node1012(X)*W[1974] + node1207(X)*W[348] + node1310(X)*W[1597]
    return(result)

@tf.function
def node1632(X):
    result = B[1632] + tf.gather(X, 317, axis=1)*W[826] + tf.gather(X, 702, axis=1)*W[1401] + tf.gather(X, 736, axis=1)*W[741] + node1637(X)*W[995]
    return(result)

@tf.function
def node1633(X):
    result = B[1633]
    return(result)

@tf.function
def node1634(X):
    result = B[1634] + node901(X)*W[1277] + node1415(X)*W[804]
    return(result)

@tf.function
def node1635(X):
    result = B[1635]
    return(result)

@tf.function
def node1636(X):
    result = B[1636] + node880(X)*W[1529] + node982(X)*W[1345]
    return(result)

@tf.function
def node1637(X):
    result = B[1637] + tf.gather(X, 688, axis=1)*W[729] + node1285(X)*W[928] + node1396(X)*W[37] + node1474(X)*W[259]
    return(result)

@tf.function
def node1638(X):
    result = B[1638]
    return(result)

@tf.function
def node1639(X):
    result = B[1639] + tf.gather(X, 401, axis=1)*W[1945]
    return(result)

@tf.function
def node1640(X):
    result = B[1640] + tf.gather(X, 223, axis=1)*W[472] + tf.gather(X, 520, axis=1)*W[1956] + node1091(X)*W[78]
    return(result)

@tf.function
def node1641(X):
    result = B[1641] + tf.gather(X, 239, axis=1)*W[1228] + tf.gather(X, 326, axis=1)*W[1334]
    return(result)

@tf.function
def node1642(X):
    result = B[1642] + tf.gather(X, 84, axis=1)*W[698] + tf.gather(X, 590, axis=1)*W[1068] + node1665(X)*W[1663]
    return(result)

@tf.function
def node1643(X):
    result = B[1643] + tf.gather(X, 167, axis=1)*W[1935] + node1131(X)*W[1049] + node1448(X)*W[1261]
    return(result)

@tf.function
def node1644(X):
    result = B[1644]
    return(result)

@tf.function
def node1645(X):
    result = B[1645] + node1728(X)*W[1517] + node1743(X)*W[186]
    return(result)

@tf.function
def node1646(X):
    result = B[1646] + node902(X)*W[1067]
    return(result)

@tf.function
def node1647(X):
    result = B[1647] + tf.gather(X, 224, axis=1)*W[1359] + node1218(X)*W[1248] + node1629(X)*W[1455] + node1656(X)*W[1007]
    return(result)

@tf.function
def node1648(X):
    result = B[1648] + tf.gather(X, 437, axis=1)*W[178] + node1213(X)*W[1299]
    return(result)

@tf.function
def node1649(X):
    result = B[1649]
    return(result)

@tf.function
def node1650(X):
    result = B[1650] + tf.gather(X, 564, axis=1)*W[750] + node1155(X)*W[876] + node1486(X)*W[1622] + node1585(X)*W[1093] + node1704(X)*W[1130]
    return(result)

@tf.function
def node1651(X):
    result = B[1651] + node1254(X)*W[1755]
    return(result)

@tf.function
def node1652(X):
    result = B[1652] + tf.gather(X, 748, axis=1)*W[71]
    return(result)

@tf.function
def node1653(X):
    result = B[1653] + tf.gather(X, 267, axis=1)*W[942]
    return(result)

@tf.function
def node1654(X):
    result = B[1654] + node1287(X)*W[1548] + node1291(X)*W[1238] + node1633(X)*W[999]
    return(result)

@tf.function
def node1655(X):
    result = B[1655] + tf.gather(X, 9, axis=1)*W[530] + tf.gather(X, 542, axis=1)*W[1106] + tf.gather(X, 570, axis=1)*W[1225] + node1140(X)*W[1764] + node1222(X)*W[714]
    return(result)

@tf.function
def node1656(X):
    result = B[1656] + tf.gather(X, 120, axis=1)*W[1698] + tf.gather(X, 656, axis=1)*W[1412]
    return(result)

@tf.function
def node1657(X):
    result = B[1657] + tf.gather(X, 650, axis=1)*W[831] + node1586(X)*W[1274] + node1766(X)*W[1369]
    return(result)

@tf.function
def node1658(X):
    result = B[1658] + node1020(X)*W[1022] + node1188(X)*W[555]
    return(result)

@tf.function
def node1659(X):
    result = B[1659]
    return(result)

@tf.function
def node1660(X):
    result = B[1660] + tf.gather(X, 321, axis=1)*W[673] + tf.gather(X, 549, axis=1)*W[1728]
    return(result)

@tf.function
def node1661(X):
    result = B[1661] + tf.gather(X, 431, axis=1)*W[721] + tf.gather(X, 486, axis=1)*W[1437] + node833(X)*W[1743] + node1076(X)*W[1719]
    return(result)

@tf.function
def node1662(X):
    result = B[1662]
    return(result)

@tf.function
def node1663(X):
    result = B[1663] + tf.gather(X, 700, axis=1)*W[662] + node908(X)*W[1985]
    return(result)

@tf.function
def node1664(X):
    result = B[1664] + tf.gather(X, 560, axis=1)*W[848] + node905(X)*W[36] + node945(X)*W[267]
    return(result)

@tf.function
def node1665(X):
    result = B[1665] + node1703(X)*W[1518]
    return(result)

@tf.function
def node1666(X):
    result = B[1666]
    return(result)

@tf.function
def node1667(X):
    result = B[1667] + tf.gather(X, 686, axis=1)*W[1572] + node1313(X)*W[713] + node1365(X)*W[660] + node1748(X)*W[1110]
    return(result)

@tf.function
def node1668(X):
    result = B[1668] + tf.gather(X, 153, axis=1)*W[1044] + tf.gather(X, 271, axis=1)*W[954] + node1790(X)*W[859]
    return(result)

@tf.function
def node1669(X):
    result = B[1669] + tf.gather(X, 687, axis=1)*W[340] + node826(X)*W[1575] + node1761(X)*W[1933]
    return(result)

@tf.function
def node1670(X):
    result = B[1670] + tf.gather(X, 420, axis=1)*W[1714]
    return(result)

@tf.function
def node1671(X):
    result = B[1671] + tf.gather(X, 131, axis=1)*W[1826] + tf.gather(X, 500, axis=1)*W[252]
    return(result)

@tf.function
def node1672(X):
    result = B[1672] + node846(X)*W[957] + node1023(X)*W[1402]
    return(result)

@tf.function
def node1673(X):
    result = B[1673] + node1066(X)*W[1381]
    return(result)

@tf.function
def node1674(X):
    result = B[1674]
    return(result)

@tf.function
def node1675(X):
    result = B[1675] + node1029(X)*W[784]
    return(result)

@tf.function
def node1676(X):
    result = B[1676] + node874(X)*W[1048]
    return(result)

@tf.function
def node1677(X):
    result = B[1677] + node924(X)*W[430] + node1271(X)*W[184] + node1309(X)*W[725]
    return(result)

@tf.function
def node1678(X):
    result = B[1678] + node972(X)*W[515]
    return(result)

@tf.function
def node1679(X):
    result = B[1679] + node1221(X)*W[52] + node1233(X)*W[707]
    return(result)

@tf.function
def node1680(X):
    result = B[1680] + tf.gather(X, 137, axis=1)*W[1825]
    return(result)

@tf.function
def node1681(X):
    result = B[1681] + tf.gather(X, 692, axis=1)*W[1573] + node985(X)*W[49] + node1383(X)*W[1636]
    return(result)

@tf.function
def node1682(X):
    result = B[1682] + node1482(X)*W[1139] + node1693(X)*W[1442]
    return(result)

@tf.function
def node1683(X):
    result = B[1683] + tf.gather(X, 157, axis=1)*W[405] + tf.gather(X, 572, axis=1)*W[645] + node937(X)*W[1658] + node1213(X)*W[1624]
    return(result)

@tf.function
def node1684(X):
    result = B[1684] + tf.gather(X, 182, axis=1)*W[1638] + node1560(X)*W[1112]
    return(result)

@tf.function
def node1685(X):
    result = B[1685] + node1465(X)*W[1947]
    return(result)

@tf.function
def node1686(X):
    result = B[1686] + tf.gather(X, 282, axis=1)*W[1056]
    return(result)

@tf.function
def node1687(X):
    result = B[1687]
    return(result)

@tf.function
def node1688(X):
    result = B[1688] + tf.gather(X, 568, axis=1)*W[947] + node912(X)*W[894] + node1184(X)*W[1024]
    return(result)

@tf.function
def node1689(X):
    result = B[1689] + tf.gather(X, 313, axis=1)*W[1907] + node1067(X)*W[1201] + node1103(X)*W[765]
    return(result)

@tf.function
def node1690(X):
    result = B[1690] + tf.gather(X, 147, axis=1)*W[529]
    return(result)

@tf.function
def node1691(X):
    result = B[1691] + node1673(X)*W[1598]
    return(result)

@tf.function
def node1692(X):
    result = B[1692]
    return(result)

@tf.function
def node1693(X):
    result = B[1693] + tf.gather(X, 139, axis=1)*W[937] + tf.gather(X, 439, axis=1)*W[384] + node961(X)*W[1842] + node1653(X)*W[1666]
    return(result)

@tf.function
def node1694(X):
    result = B[1694] + node1293(X)*W[1449] + node1569(X)*W[668]
    return(result)

@tf.function
def node1695(X):
    result = B[1695] + tf.gather(X, 393, axis=1)*W[1691]
    return(result)

@tf.function
def node1696(X):
    result = B[1696]
    return(result)

@tf.function
def node1697(X):
    result = B[1697] + tf.gather(X, 464, axis=1)*W[380]
    return(result)

@tf.function
def node1698(X):
    result = B[1698] + tf.gather(X, 463, axis=1)*W[492] + tf.gather(X, 532, axis=1)*W[860] + node1176(X)*W[715] + node1239(X)*W[1534] + node1667(X)*W[757]
    return(result)

@tf.function
def node1699(X):
    result = B[1699] + tf.gather(X, 449, axis=1)*W[782] + node1120(X)*W[763]
    return(result)

@tf.function
def node1700(X):
    result = B[1700]
    return(result)

@tf.function
def node1701(X):
    result = B[1701] + node1598(X)*W[1703]
    return(result)

@tf.function
def node1702(X):
    result = B[1702] + tf.gather(X, 216, axis=1)*W[703] + tf.gather(X, 463, axis=1)*W[1260] + node1130(X)*W[1680] + node1277(X)*W[281]
    return(result)

@tf.function
def node1703(X):
    result = B[1703]
    return(result)

@tf.function
def node1704(X):
    result = B[1704] + tf.gather(X, 194, axis=1)*W[1279] + node911(X)*W[1464] + node1400(X)*W[1654] + node1605(X)*W[1678]
    return(result)

@tf.function
def node1705(X):
    result = B[1705] + node1311(X)*W[31] + node1681(X)*W[636]
    return(result)

@tf.function
def node1706(X):
    result = B[1706] + node1401(X)*W[865]
    return(result)

@tf.function
def node1707(X):
    result = B[1707] + tf.gather(X, 36, axis=1)*W[1346]
    return(result)

@tf.function
def node1708(X):
    result = B[1708] + tf.gather(X, 217, axis=1)*W[880] + tf.gather(X, 445, axis=1)*W[1036]
    return(result)

@tf.function
def node1709(X):
    result = B[1709] + node1028(X)*W[1193] + node1145(X)*W[899] + node1491(X)*W[1734]
    return(result)

@tf.function
def node1710(X):
    result = B[1710] + tf.gather(X, 257, axis=1)*W[1473] + tf.gather(X, 421, axis=1)*W[424] + node1282(X)*W[218] + node1360(X)*W[1695]
    return(result)

@tf.function
def node1711(X):
    result = B[1711] + tf.gather(X, 546, axis=1)*W[1578]
    return(result)

@tf.function
def node1712(X):
    result = B[1712] + tf.gather(X, 169, axis=1)*W[1858] + node1736(X)*W[1204]
    return(result)

@tf.function
def node1713(X):
    result = B[1713] + tf.gather(X, 35, axis=1)*W[1057] + node1013(X)*W[368] + node1202(X)*W[389] + node1222(X)*W[210]
    return(result)

@tf.function
def node1714(X):
    result = B[1714] + node827(X)*W[1781] + node943(X)*W[1996] + node1067(X)*W[429] + node1609(X)*W[583]
    return(result)

@tf.function
def node1715(X):
    result = B[1715]
    return(result)

@tf.function
def node1716(X):
    result = B[1716] + node1548(X)*W[26]
    return(result)

@tf.function
def node1717(X):
    result = B[1717] + tf.gather(X, 305, axis=1)*W[255] + tf.gather(X, 328, axis=1)*W[275] + tf.gather(X, 542, axis=1)*W[1806] + node1553(X)*W[1811] + node1597(X)*W[1837] + node1612(X)*W[1689]
    return(result)

@tf.function
def node1718(X):
    result = B[1718]
    return(result)

@tf.function
def node1719(X):
    result = B[1719] + tf.gather(X, 229, axis=1)*W[1131] + node997(X)*W[1908] + node1552(X)*W[1630]
    return(result)

@tf.function
def node1720(X):
    result = B[1720] + tf.gather(X, 399, axis=1)*W[689] + node919(X)*W[1293] + node1019(X)*W[1104] + node1319(X)*W[1200]
    return(result)

@tf.function
def node1721(X):
    result = B[1721]
    return(result)

@tf.function
def node1722(X):
    result = B[1722] + tf.gather(X, 165, axis=1)*W[289] + node1493(X)*W[628] + node1641(X)*W[77]
    return(result)

@tf.function
def node1723(X):
    result = B[1723]
    return(result)

@tf.function
def node1724(X):
    result = B[1724] + tf.gather(X, 746, axis=1)*W[599] + node815(X)*W[1619]
    return(result)

@tf.function
def node1725(X):
    result = B[1725] + tf.gather(X, 173, axis=1)*W[45] + tf.gather(X, 283, axis=1)*W[507]
    return(result)

@tf.function
def node1726(X):
    result = B[1726] + tf.gather(X, 343, axis=1)*W[682] + node987(X)*W[639] + node1070(X)*W[1179]
    return(result)

@tf.function
def node1727(X):
    result = B[1727] + tf.gather(X, 200, axis=1)*W[595] + node934(X)*W[960]
    return(result)

@tf.function
def node1728(X):
    result = B[1728] + node1292(X)*W[1117] + node1314(X)*W[1443] + node1473(X)*W[968]
    return(result)

@tf.function
def node1729(X):
    result = B[1729] + node976(X)*W[1725] + node1665(X)*W[1465]
    return(result)

@tf.function
def node1730(X):
    result = B[1730] + node1055(X)*W[174]
    return(result)

@tf.function
def node1731(X):
    result = B[1731] + tf.gather(X, 319, axis=1)*W[1170] + tf.gather(X, 580, axis=1)*W[1335]
    return(result)

@tf.function
def node1732(X):
    result = B[1732] + tf.gather(X, 513, axis=1)*W[760]
    return(result)

@tf.function
def node1733(X):
    result = B[1733] + node1165(X)*W[1300] + node1241(X)*W[1332] + node1433(X)*W[404]
    return(result)

@tf.function
def node1734(X):
    result = B[1734]
    return(result)

@tf.function
def node1735(X):
    result = B[1735] + node1312(X)*W[597]
    return(result)

@tf.function
def node1736(X):
    result = B[1736] + tf.gather(X, 527, axis=1)*W[374] + tf.gather(X, 696, axis=1)*W[582] + node814(X)*W[345]
    return(result)

@tf.function
def node1737(X):
    result = B[1737] + tf.gather(X, 28, axis=1)*W[902] + node1518(X)*W[766]
    return(result)

@tf.function
def node1738(X):
    result = B[1738] + tf.gather(X, 102, axis=1)*W[1235] + node1197(X)*W[1136] + node1534(X)*W[866]
    return(result)

@tf.function
def node1739(X):
    result = B[1739]
    return(result)

@tf.function
def node1740(X):
    result = B[1740] + tf.gather(X, 209, axis=1)*W[998]
    return(result)

@tf.function
def node1741(X):
    result = B[1741] + tf.gather(X, 158, axis=1)*W[1386] + node858(X)*W[453]
    return(result)

@tf.function
def node1742(X):
    result = B[1742] + node1332(X)*W[261]
    return(result)

@tf.function
def node1743(X):
    result = B[1743] + tf.gather(X, 286, axis=1)*W[1774]
    return(result)

@tf.function
def node1744(X):
    result = B[1744] + node806(X)*W[1425] + node868(X)*W[1418]
    return(result)

@tf.function
def node1745(X):
    result = B[1745]
    return(result)

@tf.function
def node1746(X):
    result = B[1746] + tf.gather(X, 742, axis=1)*W[226] + node1020(X)*W[244] + node1715(X)*W[726]
    return(result)

@tf.function
def node1747(X):
    result = B[1747]
    return(result)

@tf.function
def node1748(X):
    result = B[1748] + node1014(X)*W[571] + node1779(X)*W[337]
    return(result)

@tf.function
def node1749(X):
    result = B[1749] + tf.gather(X, 137, axis=1)*W[700] + tf.gather(X, 461, axis=1)*W[575] + node1408(X)*W[780] + node1462(X)*W[1061]
    return(result)

@tf.function
def node1750(X):
    result = B[1750] + tf.gather(X, 296, axis=1)*W[131]
    return(result)

@tf.function
def node1751(X):
    result = B[1751] + tf.gather(X, 737, axis=1)*W[1522]
    return(result)

@tf.function
def node1752(X):
    result = B[1752] + node1137(X)*W[786] + node1741(X)*W[1588]
    return(result)

@tf.function
def node1753(X):
    result = B[1753] + tf.gather(X, 330, axis=1)*W[1368]
    return(result)

@tf.function
def node1754(X):
    result = B[1754] + node836(X)*W[169] + node1153(X)*W[154]
    return(result)

@tf.function
def node1755(X):
    result = B[1755] + tf.gather(X, 770, axis=1)*W[325] + node906(X)*W[1195] + node1076(X)*W[1861] + node1606(X)*W[1313]
    return(result)

@tf.function
def node1756(X):
    result = B[1756] + tf.gather(X, 41, axis=1)*W[231] + node1277(X)*W[1098] + node1606(X)*W[912]
    return(result)

@tf.function
def node1757(X):
    result = B[1757] + node984(X)*W[953] + node1382(X)*W[103]
    return(result)

@tf.function
def node1758(X):
    result = B[1758] + tf.gather(X, 471, axis=1)*W[658]
    return(result)

@tf.function
def node1759(X):
    result = B[1759] + node1546(X)*W[298]
    return(result)

@tf.function
def node1760(X):
    result = B[1760] + tf.gather(X, 271, axis=1)*W[1357] + tf.gather(X, 339, axis=1)*W[1651] + tf.gather(X, 351, axis=1)*W[17]
    return(result)

@tf.function
def node1761(X):
    result = B[1761] + node1486(X)*W[256] + node1577(X)*W[295]
    return(result)

@tf.function
def node1762(X):
    result = B[1762]
    return(result)

@tf.function
def node1763(X):
    result = B[1763]
    return(result)

@tf.function
def node1764(X):
    result = B[1764] + tf.gather(X, 156, axis=1)*W[553] + tf.gather(X, 265, axis=1)*W[878] + tf.gather(X, 659, axis=1)*W[408] + tf.gather(X, 747, axis=1)*W[487] + node1316(X)*W[1600]
    return(result)

@tf.function
def node1765(X):
    result = B[1765] + node1637(X)*W[588]
    return(result)

@tf.function
def node1766(X):
    result = B[1766]
    return(result)

@tf.function
def node1767(X):
    result = B[1767] + tf.gather(X, 205, axis=1)*W[1874] + tf.gather(X, 223, axis=1)*W[570] + node827(X)*W[215]
    return(result)

@tf.function
def node1768(X):
    result = B[1768] + node1479(X)*W[1352]
    return(result)

@tf.function
def node1769(X):
    result = B[1769]
    return(result)

@tf.function
def node1770(X):
    result = B[1770] + tf.gather(X, 110, axis=1)*W[723] + tf.gather(X, 212, axis=1)*W[276] + tf.gather(X, 651, axis=1)*W[814] + node1640(X)*W[647]
    return(result)

@tf.function
def node1771(X):
    result = B[1771]
    return(result)

@tf.function
def node1772(X):
    result = B[1772] + tf.gather(X, 280, axis=1)*W[1768] + node934(X)*W[73] + node1086(X)*W[1913]
    return(result)

@tf.function
def node1773(X):
    result = B[1773] + tf.gather(X, 687, axis=1)*W[1186]
    return(result)

@tf.function
def node1774(X):
    result = B[1774] + tf.gather(X, 317, axis=1)*W[406]
    return(result)

@tf.function
def node1775(X):
    result = B[1775]
    return(result)

@tf.function
def node1776(X):
    result = B[1776] + node1191(X)*W[1924]
    return(result)

@tf.function
def node1777(X):
    result = B[1777] + tf.gather(X, 67, axis=1)*W[981] + node1408(X)*W[652] + node1783(X)*W[1813]
    return(result)

@tf.function
def node1778(X):
    result = B[1778] + tf.gather(X, 316, axis=1)*W[1192] + node837(X)*W[1370]
    return(result)

@tf.function
def node1779(X):
    result = B[1779] + tf.gather(X, 26, axis=1)*W[219]
    return(result)

@tf.function
def node1780(X):
    result = B[1780] + node1679(X)*W[25]
    return(result)

@tf.function
def node1781(X):
    result = B[1781] + node1323(X)*W[140] + node1759(X)*W[1882]
    return(result)

@tf.function
def node1782(X):
    result = B[1782]
    return(result)

@tf.function
def node1783(X):
    result = B[1783]
    return(result)

@tf.function
def node1784(X):
    result = B[1784] + node1013(X)*W[1360] + node1246(X)*W[122] + node1255(X)*W[1087]
    return(result)

@tf.function
def node1785(X):
    result = B[1785] + tf.gather(X, 9, axis=1)*W[1671] + tf.gather(X, 750, axis=1)*W[1155]
    return(result)

@tf.function
def node1786(X):
    result = B[1786] + tf.gather(X, 560, axis=1)*W[1988] + node946(X)*W[566] + node1511(X)*W[1105]
    return(result)

@tf.function
def node1787(X):
    result = B[1787] + tf.gather(X, 598, axis=1)*W[181]
    return(result)

@tf.function
def node1788(X):
    result = B[1788]
    return(result)

@tf.function
def node1789(X):
    result = B[1789] + tf.gather(X, 544, axis=1)*W[1491] + node1573(X)*W[1223] + node1618(X)*W[320]
    return(result)

@tf.function
def node1790(X):
    result = B[1790]
    return(result)

@tf.function
def node1791(X):
    result = B[1791] + tf.gather(X, 385, axis=1)*W[258] + node865(X)*W[1302] + node880(X)*W[955] + node1252(X)*W[522]
    return(result)

@tf.function
def node1792(X):
    result = B[1792] + tf.gather(X, 217, axis=1)*W[521] + tf.gather(X, 742, axis=1)*W[473]
    return(result)

@tf.function
def node1793(X):
    result = B[1793] + node1441(X)*W[1599] + node1466(X)*W[372]
    return(result)

@tf.function
def Hypothesis(X):
    out0 = B[784] + tf.gather(X, 254, axis=1)*W[0] + tf.gather(X, 266, axis=1)*W[1521] + node1371(X)*W[945]
    out1 = B[785] + tf.gather(X, 423, axis=1)*W[1]
    out2 = B[786] + tf.gather(X, 102, axis=1)*W[1451] + tf.gather(X, 338, axis=1)*W[2] + node856(X)*W[1406] + node1206(X)*W[584] + node1219(X)*W[577]
    out3 = B[787] + tf.gather(X, 749, axis=1)*W[3] + node1394(X)*W[903] + node1710(X)*W[1701]
    out4 = B[788] + tf.gather(X, 109, axis=1)*W[4]
    out5 = B[789] + tf.gather(X, 767, axis=1)*W[5]
    out6 = B[790] + tf.gather(X, 393, axis=1)*W[6] + node872(X)*W[1831]
    out7 = B[791] + tf.gather(X, 195, axis=1)*W[7] + tf.gather(X, 461, axis=1)*W[1656] + node1574(X)*W[1501]
    out8 = B[792] + tf.gather(X, 513, axis=1)*W[8] + node1005(X)*W[806] + node1491(X)*W[1444]
    out9 = B[793] + tf.gather(X, 332, axis=1)*W[9] + node959(X)*W[621]
    result = tf.stack([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9], axis=1)
    return(result)

@tf.function
def Cost(X, Y):
    return(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hypothesis(X), labels=Y)))

def Minimize(X, Y):
    loss = lambda: Cost(X ,Y)
    tf.keras.optimizers.Adam(learning_rate).minimize(loss, [W, B])

def CorrectPrediction(X, Y):
    return(tf.equal(tf.argmax(Hypothesis(X), axis=1), tf.argmax(Y, axis=1)))

def Accuracy(X, Y):
    return(tf.reduce_mean(tf.cast(CorrectPrediction(X, Y), tf.float32)))


for epoch in range(num_epochs):
    avg_cost = 0
    num_batch = int(len(x_train) / batch_size)

    start_batch, end_batch = 0, batch_size
    cost_sum = 0
    total_acc = 0
    for i in range(num_batch):
        batch_xs, batch_ys = x_train[start_batch:end_batch], y_train[start_batch:end_batch]
        Minimize(batch_xs, batch_ys)
        cost_val = Cost(batch_xs, batch_ys)
        cost_sum += cost_val
        start_batch = start_batch + batch_size
        end_batch = end_batch + batch_size
        acc = Accuracy(batch_xs, batch_ys)
        total_acc += acc
    print('Epoch: {:04d}, Cost: {:.9f}, Acc: {:.4f}'.format(epoch + 1, cost_sum, total_acc/num_batch))
    log.write('Epoch: {:04d}, Cost: {:.9f}, Acc: {:.4f}'.format(epoch + 1, cost_sum, total_acc/num_batch))

print('Learning finished')
acc = Accuracy(x_test, y_test)
print('Accuracy = {:.4f}'.format(acc))
log.write('Accuracy = {:.4f}'.format(acc))
