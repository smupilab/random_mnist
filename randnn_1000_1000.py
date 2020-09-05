import tensorflow as tf
import random

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

print(x_train.shape)

learning_rate = 0.001
num_epochs = 20
batch_size = 100

xavier = tf.keras.initializers.GlorotUniform()

### num_inputs, num_outputs, num_hiddennodes, num_edges
### 784 10 1000 1000
W = tf.Variable(xavier([1000]))
B = tf.Variable(tf.random.normal([1794]))
@tf.function
def node794(X):
    result = B[794] + node1579(X)*W[41]
    return(result)

@tf.function
def node795(X):
    result = B[795] + tf.gather(X, 214, axis=1)*W[572]
    return(result)

@tf.function
def node796(X):
    result = B[796] + node908(X)*W[214]
    return(result)

@tf.function
def node797(X):
    result = B[797] + tf.gather(X, 665, axis=1)*W[864]
    return(result)

@tf.function
def node798(X):
    result = B[798]
    return(result)

@tf.function
def node799(X):
    result = B[799] + tf.gather(X, 272, axis=1)*W[421]
    return(result)

@tf.function
def node800(X):
    result = B[800] + node1219(X)*W[649]
    return(result)

@tf.function
def node801(X):
    result = B[801] + node1678(X)*W[674]
    return(result)

@tf.function
def node802(X):
    result = B[802] + tf.gather(X, 336, axis=1)*W[153] + node916(X)*W[289]
    return(result)

@tf.function
def node803(X):
    result = B[803] + node874(X)*W[849] + node1327(X)*W[956]
    return(result)

@tf.function
def node804(X):
    result = B[804] + node914(X)*W[293]
    return(result)

@tf.function
def node805(X):
    result = B[805] + tf.gather(X, 5, axis=1)*W[379] + tf.gather(X, 619, axis=1)*W[207]
    return(result)

@tf.function
def node806(X):
    result = B[806] + tf.gather(X, 320, axis=1)*W[533]
    return(result)

@tf.function
def node807(X):
    result = B[807]
    return(result)

@tf.function
def node808(X):
    result = B[808] + tf.gather(X, 536, axis=1)*W[751]
    return(result)

@tf.function
def node809(X):
    result = B[809]
    return(result)

@tf.function
def node810(X):
    result = B[810]
    return(result)

@tf.function
def node811(X):
    result = B[811] + tf.gather(X, 478, axis=1)*W[756] + node1685(X)*W[322]
    return(result)

@tf.function
def node812(X):
    result = B[812] + node1755(X)*W[513]
    return(result)

@tf.function
def node813(X):
    result = B[813] + tf.gather(X, 492, axis=1)*W[419] + node953(X)*W[903]
    return(result)

@tf.function
def node814(X):
    result = B[814]
    return(result)

@tf.function
def node815(X):
    result = B[815] + tf.gather(X, 564, axis=1)*W[16] + tf.gather(X, 758, axis=1)*W[312]
    return(result)

@tf.function
def node816(X):
    result = B[816]
    return(result)

@tf.function
def node817(X):
    result = B[817] + node1542(X)*W[313]
    return(result)

@tf.function
def node818(X):
    result = B[818]
    return(result)

@tf.function
def node819(X):
    result = B[819]
    return(result)

@tf.function
def node820(X):
    result = B[820]
    return(result)

@tf.function
def node821(X):
    result = B[821] + tf.gather(X, 521, axis=1)*W[803] + tf.gather(X, 700, axis=1)*W[624] + node1549(X)*W[189]
    return(result)

@tf.function
def node822(X):
    result = B[822]
    return(result)

@tf.function
def node823(X):
    result = B[823]
    return(result)

@tf.function
def node824(X):
    result = B[824] + node1081(X)*W[542]
    return(result)

@tf.function
def node825(X):
    result = B[825] + tf.gather(X, 659, axis=1)*W[964] + tf.gather(X, 765, axis=1)*W[508]
    return(result)

@tf.function
def node826(X):
    result = B[826]
    return(result)

@tf.function
def node827(X):
    result = B[827] + node811(X)*W[20] + node1039(X)*W[160] + node1188(X)*W[396]
    return(result)

@tf.function
def node828(X):
    result = B[828] + tf.gather(X, 646, axis=1)*W[928]
    return(result)

@tf.function
def node829(X):
    result = B[829] + node1254(X)*W[38] + node1316(X)*W[478]
    return(result)

@tf.function
def node830(X):
    result = B[830] + node893(X)*W[198] + node1770(X)*W[451]
    return(result)

@tf.function
def node831(X):
    result = B[831] + node838(X)*W[927]
    return(result)

@tf.function
def node832(X):
    result = B[832] + node1493(X)*W[790]
    return(result)

@tf.function
def node833(X):
    result = B[833] + tf.gather(X, 251, axis=1)*W[30] + tf.gather(X, 594, axis=1)*W[280]
    return(result)

@tf.function
def node834(X):
    result = B[834] + tf.gather(X, 742, axis=1)*W[44]
    return(result)

@tf.function
def node835(X):
    result = B[835] + node1164(X)*W[234] + node1382(X)*W[344]
    return(result)

@tf.function
def node836(X):
    result = B[836]
    return(result)

@tf.function
def node837(X):
    result = B[837]
    return(result)

@tf.function
def node838(X):
    result = B[838]
    return(result)

@tf.function
def node839(X):
    result = B[839] + node1716(X)*W[701]
    return(result)

@tf.function
def node840(X):
    result = B[840]
    return(result)

@tf.function
def node841(X):
    result = B[841]
    return(result)

@tf.function
def node842(X):
    result = B[842] + tf.gather(X, 398, axis=1)*W[683]
    return(result)

@tf.function
def node843(X):
    result = B[843] + tf.gather(X, 688, axis=1)*W[496] + node1571(X)*W[753]
    return(result)

@tf.function
def node844(X):
    result = B[844] + node1517(X)*W[534]
    return(result)

@tf.function
def node845(X):
    result = B[845] + tf.gather(X, 660, axis=1)*W[948] + node1753(X)*W[70]
    return(result)

@tf.function
def node846(X):
    result = B[846] + node1193(X)*W[731]
    return(result)

@tf.function
def node847(X):
    result = B[847]
    return(result)

@tf.function
def node848(X):
    result = B[848] + tf.gather(X, 366, axis=1)*W[877] + node1494(X)*W[544]
    return(result)

@tf.function
def node849(X):
    result = B[849] + node995(X)*W[989]
    return(result)

@tf.function
def node850(X):
    result = B[850]
    return(result)

@tf.function
def node851(X):
    result = B[851] + node1622(X)*W[668]
    return(result)

@tf.function
def node852(X):
    result = B[852] + node1037(X)*W[409] + node1770(X)*W[874]
    return(result)

@tf.function
def node853(X):
    result = B[853]
    return(result)

@tf.function
def node854(X):
    result = B[854] + node963(X)*W[427] + node1764(X)*W[529]
    return(result)

@tf.function
def node855(X):
    result = B[855] + node1036(X)*W[93] + node1292(X)*W[192]
    return(result)

@tf.function
def node856(X):
    result = B[856] + tf.gather(X, 411, axis=1)*W[33]
    return(result)

@tf.function
def node857(X):
    result = B[857] + tf.gather(X, 390, axis=1)*W[698]
    return(result)

@tf.function
def node858(X):
    result = B[858] + tf.gather(X, 0, axis=1)*W[477] + tf.gather(X, 304, axis=1)*W[461]
    return(result)

@tf.function
def node859(X):
    result = B[859] + node1114(X)*W[681] + node1189(X)*W[724]
    return(result)

@tf.function
def node860(X):
    result = B[860] + tf.gather(X, 443, axis=1)*W[963] + node956(X)*W[256]
    return(result)

@tf.function
def node861(X):
    result = B[861] + tf.gather(X, 203, axis=1)*W[31] + node1292(X)*W[949]
    return(result)

@tf.function
def node862(X):
    result = B[862] + tf.gather(X, 437, axis=1)*W[771] + node1235(X)*W[469]
    return(result)

@tf.function
def node863(X):
    result = B[863]
    return(result)

@tf.function
def node864(X):
    result = B[864] + node809(X)*W[609] + node886(X)*W[979]
    return(result)

@tf.function
def node865(X):
    result = B[865] + tf.gather(X, 59, axis=1)*W[530] + tf.gather(X, 634, axis=1)*W[170]
    return(result)

@tf.function
def node866(X):
    result = B[866] + tf.gather(X, 643, axis=1)*W[42] + node1225(X)*W[929]
    return(result)

@tf.function
def node867(X):
    result = B[867] + node952(X)*W[623] + node1163(X)*W[814] + node1624(X)*W[392]
    return(result)

@tf.function
def node868(X):
    result = B[868]
    return(result)

@tf.function
def node869(X):
    result = B[869] + tf.gather(X, 314, axis=1)*W[314] + node1425(X)*W[87]
    return(result)

@tf.function
def node870(X):
    result = B[870] + node1011(X)*W[26]
    return(result)

@tf.function
def node871(X):
    result = B[871] + node930(X)*W[362]
    return(result)

@tf.function
def node872(X):
    result = B[872]
    return(result)

@tf.function
def node873(X):
    result = B[873]
    return(result)

@tf.function
def node874(X):
    result = B[874] + node1281(X)*W[969]
    return(result)

@tf.function
def node875(X):
    result = B[875] + node1089(X)*W[986]
    return(result)

@tf.function
def node876(X):
    result = B[876] + tf.gather(X, 446, axis=1)*W[660] + node1154(X)*W[532]
    return(result)

@tf.function
def node877(X):
    result = B[877] + node1384(X)*W[134]
    return(result)

@tf.function
def node878(X):
    result = B[878] + node1518(X)*W[807]
    return(result)

@tf.function
def node879(X):
    result = B[879] + node830(X)*W[136]
    return(result)

@tf.function
def node880(X):
    result = B[880] + node1408(X)*W[28]
    return(result)

@tf.function
def node881(X):
    result = B[881] + node1628(X)*W[46]
    return(result)

@tf.function
def node882(X):
    result = B[882] + tf.gather(X, 713, axis=1)*W[708] + node1620(X)*W[747]
    return(result)

@tf.function
def node883(X):
    result = B[883] + tf.gather(X, 592, axis=1)*W[918] + tf.gather(X, 710, axis=1)*W[589] + node1662(X)*W[235]
    return(result)

@tf.function
def node884(X):
    result = B[884] + node1236(X)*W[707]
    return(result)

@tf.function
def node885(X):
    result = B[885]
    return(result)

@tf.function
def node886(X):
    result = B[886]
    return(result)

@tf.function
def node887(X):
    result = B[887]
    return(result)

@tf.function
def node888(X):
    result = B[888] + tf.gather(X, 471, axis=1)*W[121]
    return(result)

@tf.function
def node889(X):
    result = B[889] + tf.gather(X, 624, axis=1)*W[539] + tf.gather(X, 747, axis=1)*W[413]
    return(result)

@tf.function
def node890(X):
    result = B[890] + tf.gather(X, 261, axis=1)*W[780] + node1074(X)*W[813]
    return(result)

@tf.function
def node891(X):
    result = B[891]
    return(result)

@tf.function
def node892(X):
    result = B[892]
    return(result)

@tf.function
def node893(X):
    result = B[893] + tf.gather(X, 237, axis=1)*W[729]
    return(result)

@tf.function
def node894(X):
    result = B[894] + tf.gather(X, 411, axis=1)*W[171] + node852(X)*W[939]
    return(result)

@tf.function
def node895(X):
    result = B[895]
    return(result)

@tf.function
def node896(X):
    result = B[896]
    return(result)

@tf.function
def node897(X):
    result = B[897] + node919(X)*W[857] + node951(X)*W[336]
    return(result)

@tf.function
def node898(X):
    result = B[898]
    return(result)

@tf.function
def node899(X):
    result = B[899]
    return(result)

@tf.function
def node900(X):
    result = B[900] + node1139(X)*W[695]
    return(result)

@tf.function
def node901(X):
    result = B[901]
    return(result)

@tf.function
def node902(X):
    result = B[902]
    return(result)

@tf.function
def node903(X):
    result = B[903]
    return(result)

@tf.function
def node904(X):
    result = B[904] + tf.gather(X, 23, axis=1)*W[566] + node1469(X)*W[680]
    return(result)

@tf.function
def node905(X):
    result = B[905] + node893(X)*W[998]
    return(result)

@tf.function
def node906(X):
    result = B[906] + tf.gather(X, 409, axis=1)*W[339] + node1184(X)*W[140]
    return(result)

@tf.function
def node907(X):
    result = B[907]
    return(result)

@tf.function
def node908(X):
    result = B[908] + tf.gather(X, 268, axis=1)*W[642]
    return(result)

@tf.function
def node909(X):
    result = B[909] + tf.gather(X, 23, axis=1)*W[155] + node868(X)*W[870] + node1717(X)*W[205]
    return(result)

@tf.function
def node910(X):
    result = B[910] + tf.gather(X, 192, axis=1)*W[769] + tf.gather(X, 476, axis=1)*W[162] + node1283(X)*W[618]
    return(result)

@tf.function
def node911(X):
    result = B[911] + tf.gather(X, 437, axis=1)*W[522] + node1660(X)*W[233]
    return(result)

@tf.function
def node912(X):
    result = B[912] + node1699(X)*W[672]
    return(result)

@tf.function
def node913(X):
    result = B[913] + node1228(X)*W[473]
    return(result)

@tf.function
def node914(X):
    result = B[914]
    return(result)

@tf.function
def node915(X):
    result = B[915] + tf.gather(X, 386, axis=1)*W[603]
    return(result)

@tf.function
def node916(X):
    result = B[916]
    return(result)

@tf.function
def node917(X):
    result = B[917] + tf.gather(X, 454, axis=1)*W[705]
    return(result)

@tf.function
def node918(X):
    result = B[918]
    return(result)

@tf.function
def node919(X):
    result = B[919]
    return(result)

@tf.function
def node920(X):
    result = B[920] + tf.gather(X, 124, axis=1)*W[795] + node1066(X)*W[905]
    return(result)

@tf.function
def node921(X):
    result = B[921]
    return(result)

@tf.function
def node922(X):
    result = B[922] + tf.gather(X, 706, axis=1)*W[808] + node979(X)*W[209]
    return(result)

@tf.function
def node923(X):
    result = B[923] + node1451(X)*W[512] + node1502(X)*W[77]
    return(result)

@tf.function
def node924(X):
    result = B[924]
    return(result)

@tf.function
def node925(X):
    result = B[925] + node837(X)*W[151]
    return(result)

@tf.function
def node926(X):
    result = B[926]
    return(result)

@tf.function
def node927(X):
    result = B[927]
    return(result)

@tf.function
def node928(X):
    result = B[928]
    return(result)

@tf.function
def node929(X):
    result = B[929]
    return(result)

@tf.function
def node930(X):
    result = B[930] + node932(X)*W[243]
    return(result)

@tf.function
def node931(X):
    result = B[931]
    return(result)

@tf.function
def node932(X):
    result = B[932] + tf.gather(X, 80, axis=1)*W[193]
    return(result)

@tf.function
def node933(X):
    result = B[933]
    return(result)

@tf.function
def node934(X):
    result = B[934] + tf.gather(X, 575, axis=1)*W[251] + node859(X)*W[854]
    return(result)

@tf.function
def node935(X):
    result = B[935] + tf.gather(X, 565, axis=1)*W[658]
    return(result)

@tf.function
def node936(X):
    result = B[936] + node1507(X)*W[278]
    return(result)

@tf.function
def node937(X):
    result = B[937] + node923(X)*W[906]
    return(result)

@tf.function
def node938(X):
    result = B[938] + node1663(X)*W[497]
    return(result)

@tf.function
def node939(X):
    result = B[939]
    return(result)

@tf.function
def node940(X):
    result = B[940]
    return(result)

@tf.function
def node941(X):
    result = B[941] + node1030(X)*W[776]
    return(result)

@tf.function
def node942(X):
    result = B[942]
    return(result)

@tf.function
def node943(X):
    result = B[943]
    return(result)

@tf.function
def node944(X):
    result = B[944]
    return(result)

@tf.function
def node945(X):
    result = B[945]
    return(result)

@tf.function
def node946(X):
    result = B[946] + tf.gather(X, 516, axis=1)*W[443]
    return(result)

@tf.function
def node947(X):
    result = B[947] + tf.gather(X, 507, axis=1)*W[389]
    return(result)

@tf.function
def node948(X):
    result = B[948] + node1439(X)*W[872]
    return(result)

@tf.function
def node949(X):
    result = B[949] + node1281(X)*W[202] + node1693(X)*W[519]
    return(result)

@tf.function
def node950(X):
    result = B[950]
    return(result)

@tf.function
def node951(X):
    result = B[951] + tf.gather(X, 185, axis=1)*W[320] + node1310(X)*W[893]
    return(result)

@tf.function
def node952(X):
    result = B[952] + node1088(X)*W[388] + node1410(X)*W[182]
    return(result)

@tf.function
def node953(X):
    result = B[953] + node1462(X)*W[399]
    return(result)

@tf.function
def node954(X):
    result = B[954]
    return(result)

@tf.function
def node955(X):
    result = B[955] + node1254(X)*W[764]
    return(result)

@tf.function
def node956(X):
    result = B[956] + node1070(X)*W[68]
    return(result)

@tf.function
def node957(X):
    result = B[957] + node1693(X)*W[717]
    return(result)

@tf.function
def node958(X):
    result = B[958] + tf.gather(X, 722, axis=1)*W[269] + node1695(X)*W[19]
    return(result)

@tf.function
def node959(X):
    result = B[959] + tf.gather(X, 269, axis=1)*W[231]
    return(result)

@tf.function
def node960(X):
    result = B[960] + tf.gather(X, 473, axis=1)*W[296] + tf.gather(X, 505, axis=1)*W[518] + node1476(X)*W[676]
    return(result)

@tf.function
def node961(X):
    result = B[961] + node1054(X)*W[845]
    return(result)

@tf.function
def node962(X):
    result = B[962] + tf.gather(X, 465, axis=1)*W[494]
    return(result)

@tf.function
def node963(X):
    result = B[963]
    return(result)

@tf.function
def node964(X):
    result = B[964]
    return(result)

@tf.function
def node965(X):
    result = B[965]
    return(result)

@tf.function
def node966(X):
    result = B[966] + tf.gather(X, 499, axis=1)*W[400]
    return(result)

@tf.function
def node967(X):
    result = B[967] + node1061(X)*W[18]
    return(result)

@tf.function
def node968(X):
    result = B[968]
    return(result)

@tf.function
def node969(X):
    result = B[969] + tf.gather(X, 468, axis=1)*W[687] + node1238(X)*W[186]
    return(result)

@tf.function
def node970(X):
    result = B[970]
    return(result)

@tf.function
def node971(X):
    result = B[971]
    return(result)

@tf.function
def node972(X):
    result = B[972] + node1565(X)*W[944]
    return(result)

@tf.function
def node973(X):
    result = B[973]
    return(result)

@tf.function
def node974(X):
    result = B[974] + tf.gather(X, 139, axis=1)*W[510]
    return(result)

@tf.function
def node975(X):
    result = B[975] + node899(X)*W[831] + node1283(X)*W[630]
    return(result)

@tf.function
def node976(X):
    result = B[976] + tf.gather(X, 135, axis=1)*W[523] + tf.gather(X, 199, axis=1)*W[369] + node1174(X)*W[215]
    return(result)

@tf.function
def node977(X):
    result = B[977] + tf.gather(X, 137, axis=1)*W[397] + tf.gather(X, 309, axis=1)*W[288] + node887(X)*W[552] + node1508(X)*W[48] + node1517(X)*W[704]
    return(result)

@tf.function
def node978(X):
    result = B[978] + node835(X)*W[485]
    return(result)

@tf.function
def node979(X):
    result = B[979] + tf.gather(X, 18, axis=1)*W[866]
    return(result)

@tf.function
def node980(X):
    result = B[980] + node1452(X)*W[172]
    return(result)

@tf.function
def node981(X):
    result = B[981]
    return(result)

@tf.function
def node982(X):
    result = B[982] + tf.gather(X, 299, axis=1)*W[241] + node1524(X)*W[726]
    return(result)

@tf.function
def node983(X):
    result = B[983] + tf.gather(X, 467, axis=1)*W[393] + node949(X)*W[915]
    return(result)

@tf.function
def node984(X):
    result = B[984]
    return(result)

@tf.function
def node985(X):
    result = B[985]
    return(result)

@tf.function
def node986(X):
    result = B[986] + node876(X)*W[777]
    return(result)

@tf.function
def node987(X):
    result = B[987] + tf.gather(X, 245, axis=1)*W[856]
    return(result)

@tf.function
def node988(X):
    result = B[988]
    return(result)

@tf.function
def node989(X):
    result = B[989] + node1520(X)*W[117]
    return(result)

@tf.function
def node990(X):
    result = B[990]
    return(result)

@tf.function
def node991(X):
    result = B[991] + node1530(X)*W[97]
    return(result)

@tf.function
def node992(X):
    result = B[992] + tf.gather(X, 217, axis=1)*W[692]
    return(result)

@tf.function
def node993(X):
    result = B[993] + node1141(X)*W[248]
    return(result)

@tf.function
def node994(X):
    result = B[994] + node1256(X)*W[553]
    return(result)

@tf.function
def node995(X):
    result = B[995]
    return(result)

@tf.function
def node996(X):
    result = B[996]
    return(result)

@tf.function
def node997(X):
    result = B[997] + node1751(X)*W[115]
    return(result)

@tf.function
def node998(X):
    result = B[998]
    return(result)

@tf.function
def node999(X):
    result = B[999]
    return(result)

@tf.function
def node1000(X):
    result = B[1000] + tf.gather(X, 595, axis=1)*W[862] + node872(X)*W[124]
    return(result)

@tf.function
def node1001(X):
    result = B[1001]
    return(result)

@tf.function
def node1002(X):
    result = B[1002]
    return(result)

@tf.function
def node1003(X):
    result = B[1003] + tf.gather(X, 193, axis=1)*W[359]
    return(result)

@tf.function
def node1004(X):
    result = B[1004] + tf.gather(X, 11, axis=1)*W[841] + tf.gather(X, 36, axis=1)*W[174]
    return(result)

@tf.function
def node1005(X):
    result = B[1005] + node1664(X)*W[169]
    return(result)

@tf.function
def node1006(X):
    result = B[1006] + tf.gather(X, 250, axis=1)*W[239] + tf.gather(X, 363, axis=1)*W[391]
    return(result)

@tf.function
def node1007(X):
    result = B[1007]
    return(result)

@tf.function
def node1008(X):
    result = B[1008] + node1776(X)*W[959]
    return(result)

@tf.function
def node1009(X):
    result = B[1009]
    return(result)

@tf.function
def node1010(X):
    result = B[1010] + node966(X)*W[713]
    return(result)

@tf.function
def node1011(X):
    result = B[1011] + node1564(X)*W[571]
    return(result)

@tf.function
def node1012(X):
    result = B[1012] + tf.gather(X, 614, axis=1)*W[275]
    return(result)

@tf.function
def node1013(X):
    result = B[1013]
    return(result)

@tf.function
def node1014(X):
    result = B[1014] + tf.gather(X, 320, axis=1)*W[591] + node938(X)*W[912]
    return(result)

@tf.function
def node1015(X):
    result = B[1015] + node1349(X)*W[176]
    return(result)

@tf.function
def node1016(X):
    result = B[1016] + node997(X)*W[472]
    return(result)

@tf.function
def node1017(X):
    result = B[1017]
    return(result)

@tf.function
def node1018(X):
    result = B[1018]
    return(result)

@tf.function
def node1019(X):
    result = B[1019] + node909(X)*W[745]
    return(result)

@tf.function
def node1020(X):
    result = B[1020] + tf.gather(X, 189, axis=1)*W[833] + node1615(X)*W[556]
    return(result)

@tf.function
def node1021(X):
    result = B[1021]
    return(result)

@tf.function
def node1022(X):
    result = B[1022] + node919(X)*W[536]
    return(result)

@tf.function
def node1023(X):
    result = B[1023]
    return(result)

@tf.function
def node1024(X):
    result = B[1024] + node1715(X)*W[901]
    return(result)

@tf.function
def node1025(X):
    result = B[1025]
    return(result)

@tf.function
def node1026(X):
    result = B[1026] + tf.gather(X, 263, axis=1)*W[21] + node1117(X)*W[129]
    return(result)

@tf.function
def node1027(X):
    result = B[1027] + node1064(X)*W[432]
    return(result)

@tf.function
def node1028(X):
    result = B[1028]
    return(result)

@tf.function
def node1029(X):
    result = B[1029] + tf.gather(X, 766, axis=1)*W[484]
    return(result)

@tf.function
def node1030(X):
    result = B[1030] + tf.gather(X, 283, axis=1)*W[853] + tf.gather(X, 494, axis=1)*W[531]
    return(result)

@tf.function
def node1031(X):
    result = B[1031] + node1340(X)*W[340]
    return(result)

@tf.function
def node1032(X):
    result = B[1032] + tf.gather(X, 490, axis=1)*W[416] + node1264(X)*W[913] + node1413(X)*W[363]
    return(result)

@tf.function
def node1033(X):
    result = B[1033]
    return(result)

@tf.function
def node1034(X):
    result = B[1034] + tf.gather(X, 132, axis=1)*W[664] + node1378(X)*W[858]
    return(result)

@tf.function
def node1035(X):
    result = B[1035] + tf.gather(X, 487, axis=1)*W[527] + tf.gather(X, 675, axis=1)*W[199] + node1423(X)*W[613]
    return(result)

@tf.function
def node1036(X):
    result = B[1036]
    return(result)

@tf.function
def node1037(X):
    result = B[1037] + node1055(X)*W[602]
    return(result)

@tf.function
def node1038(X):
    result = B[1038] + tf.gather(X, 741, axis=1)*W[697]
    return(result)

@tf.function
def node1039(X):
    result = B[1039] + tf.gather(X, 241, axis=1)*W[737] + node1008(X)*W[871]
    return(result)

@tf.function
def node1040(X):
    result = B[1040]
    return(result)

@tf.function
def node1041(X):
    result = B[1041] + node927(X)*W[581]
    return(result)

@tf.function
def node1042(X):
    result = B[1042]
    return(result)

@tf.function
def node1043(X):
    result = B[1043]
    return(result)

@tf.function
def node1044(X):
    result = B[1044] + node1239(X)*W[951] + node1345(X)*W[911]
    return(result)

@tf.function
def node1045(X):
    result = B[1045] + tf.gather(X, 727, axis=1)*W[321]
    return(result)

@tf.function
def node1046(X):
    result = B[1046]
    return(result)

@tf.function
def node1047(X):
    result = B[1047] + node928(X)*W[32]
    return(result)

@tf.function
def node1048(X):
    result = B[1048] + node1169(X)*W[867]
    return(result)

@tf.function
def node1049(X):
    result = B[1049] + tf.gather(X, 226, axis=1)*W[627]
    return(result)

@tf.function
def node1050(X):
    result = B[1050]
    return(result)

@tf.function
def node1051(X):
    result = B[1051] + tf.gather(X, 757, axis=1)*W[299] + node1454(X)*W[319]
    return(result)

@tf.function
def node1052(X):
    result = B[1052] + tf.gather(X, 122, axis=1)*W[447]
    return(result)

@tf.function
def node1053(X):
    result = B[1053]
    return(result)

@tf.function
def node1054(X):
    result = B[1054]
    return(result)

@tf.function
def node1055(X):
    result = B[1055] + tf.gather(X, 618, axis=1)*W[608] + tf.gather(X, 659, axis=1)*W[925]
    return(result)

@tf.function
def node1056(X):
    result = B[1056] + tf.gather(X, 641, axis=1)*W[889] + node1391(X)*W[899]
    return(result)

@tf.function
def node1057(X):
    result = B[1057] + tf.gather(X, 319, axis=1)*W[815]
    return(result)

@tf.function
def node1058(X):
    result = B[1058] + tf.gather(X, 239, axis=1)*W[63] + tf.gather(X, 391, axis=1)*W[376] + node1068(X)*W[784]
    return(result)

@tf.function
def node1059(X):
    result = B[1059] + node1752(X)*W[800]
    return(result)

@tf.function
def node1060(X):
    result = B[1060] + tf.gather(X, 744, axis=1)*W[23]
    return(result)

@tf.function
def node1061(X):
    result = B[1061] + tf.gather(X, 564, axis=1)*W[222] + node817(X)*W[89]
    return(result)

@tf.function
def node1062(X):
    result = B[1062] + tf.gather(X, 65, axis=1)*W[181] + node944(X)*W[723]
    return(result)

@tf.function
def node1063(X):
    result = B[1063] + tf.gather(X, 600, axis=1)*W[809]
    return(result)

@tf.function
def node1064(X):
    result = B[1064]
    return(result)

@tf.function
def node1065(X):
    result = B[1065] + tf.gather(X, 444, axis=1)*W[225]
    return(result)

@tf.function
def node1066(X):
    result = B[1066]
    return(result)

@tf.function
def node1067(X):
    result = B[1067]
    return(result)

@tf.function
def node1068(X):
    result = B[1068] + node1055(X)*W[338] + node1090(X)*W[294] + node1313(X)*W[598]
    return(result)

@tf.function
def node1069(X):
    result = B[1069]
    return(result)

@tf.function
def node1070(X):
    result = B[1070]
    return(result)

@tf.function
def node1071(X):
    result = B[1071]
    return(result)

@tf.function
def node1072(X):
    result = B[1072]
    return(result)

@tf.function
def node1073(X):
    result = B[1073]
    return(result)

@tf.function
def node1074(X):
    result = B[1074] + tf.gather(X, 717, axis=1)*W[398]
    return(result)

@tf.function
def node1075(X):
    result = B[1075] + tf.gather(X, 215, axis=1)*W[39]
    return(result)

@tf.function
def node1076(X):
    result = B[1076]
    return(result)

@tf.function
def node1077(X):
    result = B[1077]
    return(result)

@tf.function
def node1078(X):
    result = B[1078] + tf.gather(X, 123, axis=1)*W[51] + tf.gather(X, 406, axis=1)*W[94]
    return(result)

@tf.function
def node1079(X):
    result = B[1079] + tf.gather(X, 392, axis=1)*W[47]
    return(result)

@tf.function
def node1080(X):
    result = B[1080] + node939(X)*W[360]
    return(result)

@tf.function
def node1081(X):
    result = B[1081]
    return(result)

@tf.function
def node1082(X):
    result = B[1082] + node1003(X)*W[524]
    return(result)

@tf.function
def node1083(X):
    result = B[1083] + node991(X)*W[644] + node1193(X)*W[106] + node1222(X)*W[991]
    return(result)

@tf.function
def node1084(X):
    result = B[1084] + tf.gather(X, 622, axis=1)*W[892]
    return(result)

@tf.function
def node1085(X):
    result = B[1085] + tf.gather(X, 326, axis=1)*W[50] + tf.gather(X, 655, axis=1)*W[78]
    return(result)

@tf.function
def node1086(X):
    result = B[1086] + node1440(X)*W[787]
    return(result)

@tf.function
def node1087(X):
    result = B[1087]
    return(result)

@tf.function
def node1088(X):
    result = B[1088] + tf.gather(X, 154, axis=1)*W[268] + node1699(X)*W[164]
    return(result)

@tf.function
def node1089(X):
    result = B[1089] + node1321(X)*W[60]
    return(result)

@tf.function
def node1090(X):
    result = B[1090]
    return(result)

@tf.function
def node1091(X):
    result = B[1091] + node1340(X)*W[232] + node1656(X)*W[551]
    return(result)

@tf.function
def node1092(X):
    result = B[1092]
    return(result)

@tf.function
def node1093(X):
    result = B[1093]
    return(result)

@tf.function
def node1094(X):
    result = B[1094]
    return(result)

@tf.function
def node1095(X):
    result = B[1095] + node1622(X)*W[829]
    return(result)

@tf.function
def node1096(X):
    result = B[1096] + node827(X)*W[303]
    return(result)

@tf.function
def node1097(X):
    result = B[1097] + node983(X)*W[138]
    return(result)

@tf.function
def node1098(X):
    result = B[1098] + tf.gather(X, 710, axis=1)*W[525]
    return(result)

@tf.function
def node1099(X):
    result = B[1099] + node1768(X)*W[694]
    return(result)

@tf.function
def node1100(X):
    result = B[1100] + tf.gather(X, 468, axis=1)*W[441]
    return(result)

@tf.function
def node1101(X):
    result = B[1101]
    return(result)

@tf.function
def node1102(X):
    result = B[1102]
    return(result)

@tf.function
def node1103(X):
    result = B[1103]
    return(result)

@tf.function
def node1104(X):
    result = B[1104] + node879(X)*W[702] + node1353(X)*W[972]
    return(result)

@tf.function
def node1105(X):
    result = B[1105] + node1152(X)*W[228]
    return(result)

@tf.function
def node1106(X):
    result = B[1106] + tf.gather(X, 361, axis=1)*W[468] + tf.gather(X, 408, axis=1)*W[440] + node1166(X)*W[760]
    return(result)

@tf.function
def node1107(X):
    result = B[1107]
    return(result)

@tf.function
def node1108(X):
    result = B[1108] + tf.gather(X, 516, axis=1)*W[255]
    return(result)

@tf.function
def node1109(X):
    result = B[1109] + node1716(X)*W[759]
    return(result)

@tf.function
def node1110(X):
    result = B[1110] + tf.gather(X, 240, axis=1)*W[223] + node938(X)*W[13]
    return(result)

@tf.function
def node1111(X):
    result = B[1111] + tf.gather(X, 244, axis=1)*W[503]
    return(result)

@tf.function
def node1112(X):
    result = B[1112]
    return(result)

@tf.function
def node1113(X):
    result = B[1113]
    return(result)

@tf.function
def node1114(X):
    result = B[1114]
    return(result)

@tf.function
def node1115(X):
    result = B[1115] + tf.gather(X, 140, axis=1)*W[590] + node1177(X)*W[196] + node1658(X)*W[962]
    return(result)

@tf.function
def node1116(X):
    result = B[1116] + tf.gather(X, 121, axis=1)*W[632] + tf.gather(X, 482, axis=1)*W[88] + node1370(X)*W[647]
    return(result)

@tf.function
def node1117(X):
    result = B[1117]
    return(result)

@tf.function
def node1118(X):
    result = B[1118]
    return(result)

@tf.function
def node1119(X):
    result = B[1119]
    return(result)

@tf.function
def node1120(X):
    result = B[1120] + tf.gather(X, 753, axis=1)*W[166] + node1096(X)*W[471]
    return(result)

@tf.function
def node1121(X):
    result = B[1121]
    return(result)

@tf.function
def node1122(X):
    result = B[1122] + tf.gather(X, 447, axis=1)*W[631] + node1734(X)*W[423]
    return(result)

@tf.function
def node1123(X):
    result = B[1123]
    return(result)

@tf.function
def node1124(X):
    result = B[1124] + node1212(X)*W[873]
    return(result)

@tf.function
def node1125(X):
    result = B[1125]
    return(result)

@tf.function
def node1126(X):
    result = B[1126]
    return(result)

@tf.function
def node1127(X):
    result = B[1127] + node918(X)*W[71]
    return(result)

@tf.function
def node1128(X):
    result = B[1128] + tf.gather(X, 250, axis=1)*W[826] + tf.gather(X, 371, axis=1)*W[295]
    return(result)

@tf.function
def node1129(X):
    result = B[1129] + tf.gather(X, 780, axis=1)*W[667] + node1607(X)*W[665]
    return(result)

@tf.function
def node1130(X):
    result = B[1130]
    return(result)

@tf.function
def node1131(X):
    result = B[1131] + node1324(X)*W[773]
    return(result)

@tf.function
def node1132(X):
    result = B[1132] + node1308(X)*W[640]
    return(result)

@tf.function
def node1133(X):
    result = B[1133] + tf.gather(X, 279, axis=1)*W[926] + node1369(X)*W[763]
    return(result)

@tf.function
def node1134(X):
    result = B[1134] + tf.gather(X, 352, axis=1)*W[593] + node1117(X)*W[141]
    return(result)

@tf.function
def node1135(X):
    result = B[1135]
    return(result)

@tf.function
def node1136(X):
    result = B[1136]
    return(result)

@tf.function
def node1137(X):
    result = B[1137] + tf.gather(X, 85, axis=1)*W[185] + tf.gather(X, 546, axis=1)*W[411] + tf.gather(X, 570, axis=1)*W[24]
    return(result)

@tf.function
def node1138(X):
    result = B[1138] + tf.gather(X, 332, axis=1)*W[72] + node1078(X)*W[733]
    return(result)

@tf.function
def node1139(X):
    result = B[1139]
    return(result)

@tf.function
def node1140(X):
    result = B[1140] + node1013(X)*W[120] + node1628(X)*W[967]
    return(result)

@tf.function
def node1141(X):
    result = B[1141] + tf.gather(X, 465, axis=1)*W[611] + node1722(X)*W[528]
    return(result)

@tf.function
def node1142(X):
    result = B[1142] + tf.gather(X, 267, axis=1)*W[762] + tf.gather(X, 553, axis=1)*W[112] + node1012(X)*W[628]
    return(result)

@tf.function
def node1143(X):
    result = B[1143]
    return(result)

@tf.function
def node1144(X):
    result = B[1144] + node1041(X)*W[985] + node1250(X)*W[732]
    return(result)

@tf.function
def node1145(X):
    result = B[1145] + tf.gather(X, 448, axis=1)*W[287]
    return(result)

@tf.function
def node1146(X):
    result = B[1146]
    return(result)

@tf.function
def node1147(X):
    result = B[1147] + tf.gather(X, 484, axis=1)*W[418] + node1547(X)*W[797]
    return(result)

@tf.function
def node1148(X):
    result = B[1148]
    return(result)

@tf.function
def node1149(X):
    result = B[1149] + node813(X)*W[168]
    return(result)

@tf.function
def node1150(X):
    result = B[1150] + node1513(X)*W[804]
    return(result)

@tf.function
def node1151(X):
    result = B[1151]
    return(result)

@tf.function
def node1152(X):
    result = B[1152] + node1114(X)*W[791] + node1538(X)*W[175]
    return(result)

@tf.function
def node1153(X):
    result = B[1153] + tf.gather(X, 7, axis=1)*W[844] + tf.gather(X, 666, axis=1)*W[583] + node1402(X)*W[573]
    return(result)

@tf.function
def node1154(X):
    result = B[1154] + tf.gather(X, 46, axis=1)*W[562] + node1706(X)*W[98]
    return(result)

@tf.function
def node1155(X):
    result = B[1155] + tf.gather(X, 319, axis=1)*W[908] + node1703(X)*W[576]
    return(result)

@tf.function
def node1156(X):
    result = B[1156] + tf.gather(X, 231, axis=1)*W[346] + node1363(X)*W[345]
    return(result)

@tf.function
def node1157(X):
    result = B[1157] + node1176(X)*W[599]
    return(result)

@tf.function
def node1158(X):
    result = B[1158] + tf.gather(X, 395, axis=1)*W[897] + tf.gather(X, 609, axis=1)*W[586] + node960(X)*W[123]
    return(result)

@tf.function
def node1159(X):
    result = B[1159] + tf.gather(X, 689, axis=1)*W[111]
    return(result)

@tf.function
def node1160(X):
    result = B[1160] + tf.gather(X, 541, axis=1)*W[565]
    return(result)

@tf.function
def node1161(X):
    result = B[1161] + tf.gather(X, 66, axis=1)*W[755] + node1283(X)*W[778]
    return(result)

@tf.function
def node1162(X):
    result = B[1162] + tf.gather(X, 54, axis=1)*W[868]
    return(result)

@tf.function
def node1163(X):
    result = B[1163] + tf.gather(X, 610, axis=1)*W[483] + node922(X)*W[457]
    return(result)

@tf.function
def node1164(X):
    result = B[1164] + node1304(X)*W[331]
    return(result)

@tf.function
def node1165(X):
    result = B[1165]
    return(result)

@tf.function
def node1166(X):
    result = B[1166] + node977(X)*W[922]
    return(result)

@tf.function
def node1167(X):
    result = B[1167] + tf.gather(X, 196, axis=1)*W[76] + tf.gather(X, 749, axis=1)*W[332]
    return(result)

@tf.function
def node1168(X):
    result = B[1168] + tf.gather(X, 350, axis=1)*W[401] + node1289(X)*W[227]
    return(result)

@tf.function
def node1169(X):
    result = B[1169] + tf.gather(X, 404, axis=1)*W[932] + tf.gather(X, 568, axis=1)*W[266]
    return(result)

@tf.function
def node1170(X):
    result = B[1170]
    return(result)

@tf.function
def node1171(X):
    result = B[1171] + tf.gather(X, 42, axis=1)*W[788]
    return(result)

@tf.function
def node1172(X):
    result = B[1172] + tf.gather(X, 502, axis=1)*W[846]
    return(result)

@tf.function
def node1173(X):
    result = B[1173] + tf.gather(X, 49, axis=1)*W[213] + tf.gather(X, 319, axis=1)*W[654]
    return(result)

@tf.function
def node1174(X):
    result = B[1174]
    return(result)

@tf.function
def node1175(X):
    result = B[1175] + tf.gather(X, 522, axis=1)*W[735]
    return(result)

@tf.function
def node1176(X):
    result = B[1176] + node981(X)*W[349]
    return(result)

@tf.function
def node1177(X):
    result = B[1177]
    return(result)

@tf.function
def node1178(X):
    result = B[1178] + tf.gather(X, 684, axis=1)*W[431]
    return(result)

@tf.function
def node1179(X):
    result = B[1179] + tf.gather(X, 428, axis=1)*W[935] + tf.gather(X, 451, axis=1)*W[699]
    return(result)

@tf.function
def node1180(X):
    result = B[1180]
    return(result)

@tf.function
def node1181(X):
    result = B[1181]
    return(result)

@tf.function
def node1182(X):
    result = B[1182]
    return(result)

@tf.function
def node1183(X):
    result = B[1183] + tf.gather(X, 511, axis=1)*W[768]
    return(result)

@tf.function
def node1184(X):
    result = B[1184] + tf.gather(X, 272, axis=1)*W[952] + node1521(X)*W[588]
    return(result)

@tf.function
def node1185(X):
    result = B[1185]
    return(result)

@tf.function
def node1186(X):
    result = B[1186] + tf.gather(X, 141, axis=1)*W[377] + tf.gather(X, 228, axis=1)*W[118] + tf.gather(X, 755, axis=1)*W[290]
    return(result)

@tf.function
def node1187(X):
    result = B[1187]
    return(result)

@tf.function
def node1188(X):
    result = B[1188]
    return(result)

@tf.function
def node1189(X):
    result = B[1189]
    return(result)

@tf.function
def node1190(X):
    result = B[1190]
    return(result)

@tf.function
def node1191(X):
    result = B[1191] + tf.gather(X, 299, axis=1)*W[700]
    return(result)

@tf.function
def node1192(X):
    result = B[1192]
    return(result)

@tf.function
def node1193(X):
    result = B[1193] + tf.gather(X, 723, axis=1)*W[617]
    return(result)

@tf.function
def node1194(X):
    result = B[1194] + node1111(X)*W[380]
    return(result)

@tf.function
def node1195(X):
    result = B[1195] + node833(X)*W[144]
    return(result)

@tf.function
def node1196(X):
    result = B[1196] + tf.gather(X, 46, axis=1)*W[394] + tf.gather(X, 746, axis=1)*W[14]
    return(result)

@tf.function
def node1197(X):
    result = B[1197] + node946(X)*W[869] + node1675(X)*W[81]
    return(result)

@tf.function
def node1198(X):
    result = B[1198]
    return(result)

@tf.function
def node1199(X):
    result = B[1199]
    return(result)

@tf.function
def node1200(X):
    result = B[1200]
    return(result)

@tf.function
def node1201(X):
    result = B[1201] + tf.gather(X, 185, axis=1)*W[58]
    return(result)

@tf.function
def node1202(X):
    result = B[1202] + node1009(X)*W[806] + node1283(X)*W[390]
    return(result)

@tf.function
def node1203(X):
    result = B[1203] + tf.gather(X, 626, axis=1)*W[798]
    return(result)

@tf.function
def node1204(X):
    result = B[1204]
    return(result)

@tf.function
def node1205(X):
    result = B[1205] + node1080(X)*W[706]
    return(result)

@tf.function
def node1206(X):
    result = B[1206] + tf.gather(X, 324, axis=1)*W[968]
    return(result)

@tf.function
def node1207(X):
    result = B[1207] + tf.gather(X, 681, axis=1)*W[884] + node940(X)*W[499] + node1231(X)*W[535] + node1459(X)*W[615] + node1652(X)*W[367]
    return(result)

@tf.function
def node1208(X):
    result = B[1208] + node1074(X)*W[90] + node1687(X)*W[184]
    return(result)

@tf.function
def node1209(X):
    result = B[1209] + node1516(X)*W[109]
    return(result)

@tf.function
def node1210(X):
    result = B[1210] + node1569(X)*W[775]
    return(result)

@tf.function
def node1211(X):
    result = B[1211] + tf.gather(X, 343, axis=1)*W[10]
    return(result)

@tf.function
def node1212(X):
    result = B[1212]
    return(result)

@tf.function
def node1213(X):
    result = B[1213]
    return(result)

@tf.function
def node1214(X):
    result = B[1214] + node1098(X)*W[422] + node1666(X)*W[387]
    return(result)

@tf.function
def node1215(X):
    result = B[1215] + tf.gather(X, 773, axis=1)*W[402] + node930(X)*W[563] + node1021(X)*W[691] + node1110(X)*W[358] + node1775(X)*W[900]
    return(result)

@tf.function
def node1216(X):
    result = B[1216] + node965(X)*W[261]
    return(result)

@tf.function
def node1217(X):
    result = B[1217]
    return(result)

@tf.function
def node1218(X):
    result = B[1218] + node1650(X)*W[550]
    return(result)

@tf.function
def node1219(X):
    result = B[1219] + node1755(X)*W[567]
    return(result)

@tf.function
def node1220(X):
    result = B[1220] + node1429(X)*W[217]
    return(result)

@tf.function
def node1221(X):
    result = B[1221] + node1666(X)*W[372]
    return(result)

@tf.function
def node1222(X):
    result = B[1222]
    return(result)

@tf.function
def node1223(X):
    result = B[1223]
    return(result)

@tf.function
def node1224(X):
    result = B[1224]
    return(result)

@tf.function
def node1225(X):
    result = B[1225]
    return(result)

@tf.function
def node1226(X):
    result = B[1226]
    return(result)

@tf.function
def node1227(X):
    result = B[1227] + node1071(X)*W[361]
    return(result)

@tf.function
def node1228(X):
    result = B[1228] + node1254(X)*W[101]
    return(result)

@tf.function
def node1229(X):
    result = B[1229]
    return(result)

@tf.function
def node1230(X):
    result = B[1230] + tf.gather(X, 776, axis=1)*W[742] + node1108(X)*W[244] + node1456(X)*W[569]
    return(result)

@tf.function
def node1231(X):
    result = B[1231] + tf.gather(X, 634, axis=1)*W[79]
    return(result)

@tf.function
def node1232(X):
    result = B[1232] + tf.gather(X, 638, axis=1)*W[652]
    return(result)

@tf.function
def node1233(X):
    result = B[1233]
    return(result)

@tf.function
def node1234(X):
    result = B[1234]
    return(result)

@tf.function
def node1235(X):
    result = B[1235]
    return(result)

@tf.function
def node1236(X):
    result = B[1236] + tf.gather(X, 190, axis=1)*W[307] + tf.gather(X, 276, axis=1)*W[64]
    return(result)

@tf.function
def node1237(X):
    result = B[1237] + node1595(X)*W[341]
    return(result)

@tf.function
def node1238(X):
    result = B[1238]
    return(result)

@tf.function
def node1239(X):
    result = B[1239] + node1280(X)*W[310]
    return(result)

@tf.function
def node1240(X):
    result = B[1240] + tf.gather(X, 685, axis=1)*W[305]
    return(result)

@tf.function
def node1241(X):
    result = B[1241]
    return(result)

@tf.function
def node1242(X):
    result = B[1242]
    return(result)

@tf.function
def node1243(X):
    result = B[1243] + tf.gather(X, 493, axis=1)*W[201]
    return(result)

@tf.function
def node1244(X):
    result = B[1244] + node1288(X)*W[466]
    return(result)

@tf.function
def node1245(X):
    result = B[1245] + tf.gather(X, 49, axis=1)*W[316]
    return(result)

@tf.function
def node1246(X):
    result = B[1246] + node1503(X)*W[479]
    return(result)

@tf.function
def node1247(X):
    result = B[1247]
    return(result)

@tf.function
def node1248(X):
    result = B[1248] + node1214(X)*W[412] + node1689(X)*W[601]
    return(result)

@tf.function
def node1249(X):
    result = B[1249]
    return(result)

@tf.function
def node1250(X):
    result = B[1250]
    return(result)

@tf.function
def node1251(X):
    result = B[1251] + tf.gather(X, 274, axis=1)*W[300] + tf.gather(X, 430, axis=1)*W[126]
    return(result)

@tf.function
def node1252(X):
    result = B[1252] + tf.gather(X, 23, axis=1)*W[317]
    return(result)

@tf.function
def node1253(X):
    result = B[1253]
    return(result)

@tf.function
def node1254(X):
    result = B[1254] + node1184(X)*W[502]
    return(result)

@tf.function
def node1255(X):
    result = B[1255] + tf.gather(X, 470, axis=1)*W[458]
    return(result)

@tf.function
def node1256(X):
    result = B[1256] + node1180(X)*W[914] + node1329(X)*W[298]
    return(result)

@tf.function
def node1257(X):
    result = B[1257] + node1136(X)*W[580]
    return(result)

@tf.function
def node1258(X):
    result = B[1258] + tf.gather(X, 333, axis=1)*W[491] + tf.gather(X, 755, axis=1)*W[99]
    return(result)

@tf.function
def node1259(X):
    result = B[1259]
    return(result)

@tf.function
def node1260(X):
    result = B[1260]
    return(result)

@tf.function
def node1261(X):
    result = B[1261] + tf.gather(X, 376, axis=1)*W[154]
    return(result)

@tf.function
def node1262(X):
    result = B[1262] + tf.gather(X, 401, axis=1)*W[347] + tf.gather(X, 726, axis=1)*W[334]
    return(result)

@tf.function
def node1263(X):
    result = B[1263]
    return(result)

@tf.function
def node1264(X):
    result = B[1264]
    return(result)

@tf.function
def node1265(X):
    result = B[1265] + tf.gather(X, 555, axis=1)*W[444] + node1787(X)*W[148]
    return(result)

@tf.function
def node1266(X):
    result = B[1266]
    return(result)

@tf.function
def node1267(X):
    result = B[1267] + node1285(X)*W[116]
    return(result)

@tf.function
def node1268(X):
    result = B[1268]
    return(result)

@tf.function
def node1269(X):
    result = B[1269] + node1459(X)*W[946]
    return(result)

@tf.function
def node1270(X):
    result = B[1270] + node1745(X)*W[973]
    return(result)

@tf.function
def node1271(X):
    result = B[1271]
    return(result)

@tf.function
def node1272(X):
    result = B[1272] + node1239(X)*W[987]
    return(result)

@tf.function
def node1273(X):
    result = B[1273]
    return(result)

@tf.function
def node1274(X):
    result = B[1274]
    return(result)

@tf.function
def node1275(X):
    result = B[1275] + tf.gather(X, 384, axis=1)*W[84] + node1398(X)*W[947]
    return(result)

@tf.function
def node1276(X):
    result = B[1276] + node1150(X)*W[607]
    return(result)

@tf.function
def node1277(X):
    result = B[1277]
    return(result)

@tf.function
def node1278(X):
    result = B[1278] + node854(X)*W[230] + node996(X)*W[555] + node1685(X)*W[211]
    return(result)

@tf.function
def node1279(X):
    result = B[1279] + node1295(X)*W[95]
    return(result)

@tf.function
def node1280(X):
    result = B[1280] + node1161(X)*W[662]
    return(result)

@tf.function
def node1281(X):
    result = B[1281]
    return(result)

@tf.function
def node1282(X):
    result = B[1282]
    return(result)

@tf.function
def node1283(X):
    result = B[1283] + node1760(X)*W[774]
    return(result)

@tf.function
def node1284(X):
    result = B[1284] + tf.gather(X, 463, axis=1)*W[490] + node1461(X)*W[395]
    return(result)

@tf.function
def node1285(X):
    result = B[1285] + tf.gather(X, 657, axis=1)*W[863]
    return(result)

@tf.function
def node1286(X):
    result = B[1286] + node1521(X)*W[792]
    return(result)

@tf.function
def node1287(X):
    result = B[1287] + tf.gather(X, 628, axis=1)*W[995]
    return(result)

@tf.function
def node1288(X):
    result = B[1288] + tf.gather(X, 452, axis=1)*W[114] + node1620(X)*W[180]
    return(result)

@tf.function
def node1289(X):
    result = B[1289] + node1002(X)*W[145]
    return(result)

@tf.function
def node1290(X):
    result = B[1290] + tf.gather(X, 362, axis=1)*W[945]
    return(result)

@tf.function
def node1291(X):
    result = B[1291] + tf.gather(X, 238, axis=1)*W[135] + tf.gather(X, 531, axis=1)*W[978] + node878(X)*W[128] + node1572(X)*W[384]
    return(result)

@tf.function
def node1292(X):
    result = B[1292] + node966(X)*W[772]
    return(result)

@tf.function
def node1293(X):
    result = B[1293]
    return(result)

@tf.function
def node1294(X):
    result = B[1294] + node857(X)*W[637] + node1174(X)*W[916]
    return(result)

@tf.function
def node1295(X):
    result = B[1295]
    return(result)

@tf.function
def node1296(X):
    result = B[1296]
    return(result)

@tf.function
def node1297(X):
    result = B[1297] + tf.gather(X, 549, axis=1)*W[414]
    return(result)

@tf.function
def node1298(X):
    result = B[1298]
    return(result)

@tf.function
def node1299(X):
    result = B[1299] + tf.gather(X, 349, axis=1)*W[545]
    return(result)

@tf.function
def node1300(X):
    result = B[1300] + tf.gather(X, 382, axis=1)*W[270] + node1087(X)*W[793] + node1516(X)*W[163]
    return(result)

@tf.function
def node1301(X):
    result = B[1301] + node1198(X)*W[464] + node1395(X)*W[719] + node1627(X)*W[254]
    return(result)

@tf.function
def node1302(X):
    result = B[1302]
    return(result)

@tf.function
def node1303(X):
    result = B[1303]
    return(result)

@tf.function
def node1304(X):
    result = B[1304] + node863(X)*W[537]
    return(result)

@tf.function
def node1305(X):
    result = B[1305] + tf.gather(X, 618, axis=1)*W[224] + node1358(X)*W[930]
    return(result)

@tf.function
def node1306(X):
    result = B[1306]
    return(result)

@tf.function
def node1307(X):
    result = B[1307] + node1264(X)*W[811]
    return(result)

@tf.function
def node1308(X):
    result = B[1308] + tf.gather(X, 30, axis=1)*W[997]
    return(result)

@tf.function
def node1309(X):
    result = B[1309] + tf.gather(X, 288, axis=1)*W[638]
    return(result)

@tf.function
def node1310(X):
    result = B[1310] + node1256(X)*W[917]
    return(result)

@tf.function
def node1311(X):
    result = B[1311]
    return(result)

@tf.function
def node1312(X):
    result = B[1312] + node1026(X)*W[686]
    return(result)

@tf.function
def node1313(X):
    result = B[1313]
    return(result)

@tf.function
def node1314(X):
    result = B[1314] + tf.gather(X, 409, axis=1)*W[629] + node1152(X)*W[492]
    return(result)

@tf.function
def node1315(X):
    result = B[1315] + node1209(X)*W[131]
    return(result)

@tf.function
def node1316(X):
    result = B[1316] + tf.gather(X, 727, axis=1)*W[283] + node1404(X)*W[505]
    return(result)

@tf.function
def node1317(X):
    result = B[1317]
    return(result)

@tf.function
def node1318(X):
    result = B[1318] + tf.gather(X, 631, axis=1)*W[585]
    return(result)

@tf.function
def node1319(X):
    result = B[1319] + tf.gather(X, 71, axis=1)*W[937] + tf.gather(X, 119, axis=1)*W[282] + node1096(X)*W[456] + node1139(X)*W[538]
    return(result)

@tf.function
def node1320(X):
    result = B[1320]
    return(result)

@tf.function
def node1321(X):
    result = B[1321] + tf.gather(X, 683, axis=1)*W[237]
    return(result)

@tf.function
def node1322(X):
    result = B[1322] + node914(X)*W[720] + node1160(X)*W[504]
    return(result)

@tf.function
def node1323(X):
    result = B[1323]
    return(result)

@tf.function
def node1324(X):
    result = B[1324] + node1076(X)*W[568]
    return(result)

@tf.function
def node1325(X):
    result = B[1325]
    return(result)

@tf.function
def node1326(X):
    result = B[1326] + node1215(X)*W[626]
    return(result)

@tf.function
def node1327(X):
    result = B[1327]
    return(result)

@tf.function
def node1328(X):
    result = B[1328] + node1466(X)*W[187]
    return(result)

@tf.function
def node1329(X):
    result = B[1329]
    return(result)

@tf.function
def node1330(X):
    result = B[1330] + tf.gather(X, 492, axis=1)*W[663] + node1572(X)*W[475]
    return(result)

@tf.function
def node1331(X):
    result = B[1331]
    return(result)

@tf.function
def node1332(X):
    result = B[1332]
    return(result)

@tf.function
def node1333(X):
    result = B[1333] + node830(X)*W[744] + node1473(X)*W[480]
    return(result)

@tf.function
def node1334(X):
    result = B[1334] + node1597(X)*W[318]
    return(result)

@tf.function
def node1335(X):
    result = B[1335] + node1399(X)*W[734]
    return(result)

@tf.function
def node1336(X):
    result = B[1336] + tf.gather(X, 439, axis=1)*W[11] + node821(X)*W[326]
    return(result)

@tf.function
def node1337(X):
    result = B[1337] + node920(X)*W[817]
    return(result)

@tf.function
def node1338(X):
    result = B[1338] + node1777(X)*W[56]
    return(result)

@tf.function
def node1339(X):
    result = B[1339] + node1495(X)*W[738]
    return(result)

@tf.function
def node1340(X):
    result = B[1340]
    return(result)

@tf.function
def node1341(X):
    result = B[1341]
    return(result)

@tf.function
def node1342(X):
    result = B[1342] + node861(X)*W[113] + node1175(X)*W[149] + node1481(X)*W[883]
    return(result)

@tf.function
def node1343(X):
    result = B[1343] + node952(X)*W[961] + node1121(X)*W[511] + node1682(X)*W[558]
    return(result)

@tf.function
def node1344(X):
    result = B[1344]
    return(result)

@tf.function
def node1345(X):
    result = B[1345]
    return(result)

@tf.function
def node1346(X):
    result = B[1346] + node1262(X)*W[643]
    return(result)

@tf.function
def node1347(X):
    result = B[1347]
    return(result)

@tf.function
def node1348(X):
    result = B[1348] + node1382(X)*W[606] + node1450(X)*W[125]
    return(result)

@tf.function
def node1349(X):
    result = B[1349]
    return(result)

@tf.function
def node1350(X):
    result = B[1350] + tf.gather(X, 225, axis=1)*W[876] + node1394(X)*W[848]
    return(result)

@tf.function
def node1351(X):
    result = B[1351] + node955(X)*W[810] + node1121(X)*W[955]
    return(result)

@tf.function
def node1352(X):
    result = B[1352] + tf.gather(X, 733, axis=1)*W[600]
    return(result)

@tf.function
def node1353(X):
    result = B[1353] + node1243(X)*W[61] + node1749(X)*W[152]
    return(result)

@tf.function
def node1354(X):
    result = B[1354] + node1180(X)*W[500]
    return(result)

@tf.function
def node1355(X):
    result = B[1355] + tf.gather(X, 724, axis=1)*W[488]
    return(result)

@tf.function
def node1356(X):
    result = B[1356] + node1254(X)*W[43]
    return(result)

@tf.function
def node1357(X):
    result = B[1357] + node1036(X)*W[108]
    return(result)

@tf.function
def node1358(X):
    result = B[1358]
    return(result)

@tf.function
def node1359(X):
    result = B[1359] + tf.gather(X, 635, axis=1)*W[105]
    return(result)

@tf.function
def node1360(X):
    result = B[1360]
    return(result)

@tf.function
def node1361(X):
    result = B[1361]
    return(result)

@tf.function
def node1362(X):
    result = B[1362] + tf.gather(X, 620, axis=1)*W[584]
    return(result)

@tf.function
def node1363(X):
    result = B[1363]
    return(result)

@tf.function
def node1364(X):
    result = B[1364]
    return(result)

@tf.function
def node1365(X):
    result = B[1365] + tf.gather(X, 98, axis=1)*W[710]
    return(result)

@tf.function
def node1366(X):
    result = B[1366]
    return(result)

@tf.function
def node1367(X):
    result = B[1367]
    return(result)

@tf.function
def node1368(X):
    result = B[1368]
    return(result)

@tf.function
def node1369(X):
    result = B[1369] + node976(X)*W[459] + node1379(X)*W[645]
    return(result)

@tf.function
def node1370(X):
    result = B[1370]
    return(result)

@tf.function
def node1371(X):
    result = B[1371] + tf.gather(X, 52, axis=1)*W[739] + tf.gather(X, 72, axis=1)*W[965]
    return(result)

@tf.function
def node1372(X):
    result = B[1372]
    return(result)

@tf.function
def node1373(X):
    result = B[1373] + tf.gather(X, 560, axis=1)*W[271]
    return(result)

@tf.function
def node1374(X):
    result = B[1374]
    return(result)

@tf.function
def node1375(X):
    result = B[1375] + tf.gather(X, 634, axis=1)*W[779]
    return(result)

@tf.function
def node1376(X):
    result = B[1376] + tf.gather(X, 80, axis=1)*W[677] + tf.gather(X, 138, axis=1)*W[746] + tf.gather(X, 549, axis=1)*W[752] + tf.gather(X, 627, axis=1)*W[992]
    return(result)

@tf.function
def node1377(X):
    result = B[1377] + node1570(X)*W[404]
    return(result)

@tf.function
def node1378(X):
    result = B[1378] + node1537(X)*W[226] + node1655(X)*W[661]
    return(result)

@tf.function
def node1379(X):
    result = B[1379]
    return(result)

@tf.function
def node1380(X):
    result = B[1380] + node847(X)*W[351] + node1456(X)*W[159]
    return(result)

@tf.function
def node1381(X):
    result = B[1381] + tf.gather(X, 377, axis=1)*W[983]
    return(result)

@tf.function
def node1382(X):
    result = B[1382] + node1086(X)*W[150]
    return(result)

@tf.function
def node1383(X):
    result = B[1383]
    return(result)

@tf.function
def node1384(X):
    result = B[1384] + tf.gather(X, 436, axis=1)*W[177] + tf.gather(X, 704, axis=1)*W[329] + tf.gather(X, 707, axis=1)*W[179] + node1315(X)*W[212]
    return(result)

@tf.function
def node1385(X):
    result = B[1385] + node1493(X)*W[796] + node1708(X)*W[950]
    return(result)

@tf.function
def node1386(X):
    result = B[1386] + node1603(X)*W[622]
    return(result)

@tf.function
def node1387(X):
    result = B[1387] + tf.gather(X, 363, axis=1)*W[374] + tf.gather(X, 406, axis=1)*W[262]
    return(result)

@tf.function
def node1388(X):
    result = B[1388]
    return(result)

@tf.function
def node1389(X):
    result = B[1389]
    return(result)

@tf.function
def node1390(X):
    result = B[1390] + tf.gather(X, 535, axis=1)*W[679]
    return(result)

@tf.function
def node1391(X):
    result = B[1391] + tf.gather(X, 402, axis=1)*W[828] + tf.gather(X, 443, axis=1)*W[921]
    return(result)

@tf.function
def node1392(X):
    result = B[1392] + tf.gather(X, 71, axis=1)*W[988]
    return(result)

@tf.function
def node1393(X):
    result = B[1393]
    return(result)

@tf.function
def node1394(X):
    result = B[1394] + node840(X)*W[15] + node1363(X)*W[758] + node1658(X)*W[721]
    return(result)

@tf.function
def node1395(X):
    result = B[1395]
    return(result)

@tf.function
def node1396(X):
    result = B[1396] + tf.gather(X, 412, axis=1)*W[29] + node850(X)*W[45] + node1546(X)*W[718]
    return(result)

@tf.function
def node1397(X):
    result = B[1397]
    return(result)

@tf.function
def node1398(X):
    result = B[1398]
    return(result)

@tf.function
def node1399(X):
    result = B[1399]
    return(result)

@tf.function
def node1400(X):
    result = B[1400]
    return(result)

@tf.function
def node1401(X):
    result = B[1401] + node1369(X)*W[669]
    return(result)

@tf.function
def node1402(X):
    result = B[1402]
    return(result)

@tf.function
def node1403(X):
    result = B[1403]
    return(result)

@tf.function
def node1404(X):
    result = B[1404] + tf.gather(X, 6, axis=1)*W[122] + tf.gather(X, 517, axis=1)*W[880]
    return(result)

@tf.function
def node1405(X):
    result = B[1405] + node1494(X)*W[633]
    return(result)

@tf.function
def node1406(X):
    result = B[1406] + node1370(X)*W[304]
    return(result)

@tf.function
def node1407(X):
    result = B[1407] + node1299(X)*W[210] + node1628(X)*W[725]
    return(result)

@tf.function
def node1408(X):
    result = B[1408] + tf.gather(X, 484, axis=1)*W[328]
    return(result)

@tf.function
def node1409(X):
    result = B[1409] + tf.gather(X, 225, axis=1)*W[386] + tf.gather(X, 515, axis=1)*W[277]
    return(result)

@tf.function
def node1410(X):
    result = B[1410]
    return(result)

@tf.function
def node1411(X):
    result = B[1411] + tf.gather(X, 467, axis=1)*W[625] + node1776(X)*W[514]
    return(result)

@tf.function
def node1412(X):
    result = B[1412] + node1179(X)*W[582]
    return(result)

@tf.function
def node1413(X):
    result = B[1413] + tf.gather(X, 108, axis=1)*W[650] + tf.gather(X, 384, axis=1)*W[337] + node1595(X)*W[895] + node1723(X)*W[450]
    return(result)

@tf.function
def node1414(X):
    result = B[1414] + node1445(X)*W[188]
    return(result)

@tf.function
def node1415(X):
    result = B[1415] + node1099(X)*W[620] + node1191(X)*W[827]
    return(result)

@tf.function
def node1416(X):
    result = B[1416] + tf.gather(X, 262, axis=1)*W[259] + node1600(X)*W[375]
    return(result)

@tf.function
def node1417(X):
    result = B[1417] + node1435(X)*W[342]
    return(result)

@tf.function
def node1418(X):
    result = B[1418]
    return(result)

@tf.function
def node1419(X):
    result = B[1419] + tf.gather(X, 310, axis=1)*W[636] + node1072(X)*W[206]
    return(result)

@tf.function
def node1420(X):
    result = B[1420] + tf.gather(X, 329, axis=1)*W[476] + tf.gather(X, 475, axis=1)*W[675]
    return(result)

@tf.function
def node1421(X):
    result = B[1421] + tf.gather(X, 732, axis=1)*W[971]
    return(result)

@tf.function
def node1422(X):
    result = B[1422] + tf.gather(X, 88, axis=1)*W[309] + node1762(X)*W[242]
    return(result)

@tf.function
def node1423(X):
    result = B[1423]
    return(result)

@tf.function
def node1424(X):
    result = B[1424]
    return(result)

@tf.function
def node1425(X):
    result = B[1425]
    return(result)

@tf.function
def node1426(X):
    result = B[1426]
    return(result)

@tf.function
def node1427(X):
    result = B[1427] + node1268(X)*W[107]
    return(result)

@tf.function
def node1428(X):
    result = B[1428]
    return(result)

@tf.function
def node1429(X):
    result = B[1429] + tf.gather(X, 334, axis=1)*W[385] + node1770(X)*W[218]
    return(result)

@tf.function
def node1430(X):
    result = B[1430]
    return(result)

@tf.function
def node1431(X):
    result = B[1431]
    return(result)

@tf.function
def node1432(X):
    result = B[1432] + tf.gather(X, 473, axis=1)*W[325]
    return(result)

@tf.function
def node1433(X):
    result = B[1433]
    return(result)

@tf.function
def node1434(X):
    result = B[1434]
    return(result)

@tf.function
def node1435(X):
    result = B[1435] + node866(X)*W[66]
    return(result)

@tf.function
def node1436(X):
    result = B[1436] + tf.gather(X, 447, axis=1)*W[886]
    return(result)

@tf.function
def node1437(X):
    result = B[1437]
    return(result)

@tf.function
def node1438(X):
    result = B[1438]
    return(result)

@tf.function
def node1439(X):
    result = B[1439] + node955(X)*W[940] + node1132(X)*W[728]
    return(result)

@tf.function
def node1440(X):
    result = B[1440] + node1519(X)*W[909]
    return(result)

@tf.function
def node1441(X):
    result = B[1441]
    return(result)

@tf.function
def node1442(X):
    result = B[1442]
    return(result)

@tf.function
def node1443(X):
    result = B[1443] + node1711(X)*W[55]
    return(result)

@tf.function
def node1444(X):
    result = B[1444] + tf.gather(X, 185, axis=1)*W[455] + node884(X)*W[240] + node893(X)*W[267]
    return(result)

@tf.function
def node1445(X):
    result = B[1445]
    return(result)

@tf.function
def node1446(X):
    result = B[1446]
    return(result)

@tf.function
def node1447(X):
    result = B[1447] + tf.gather(X, 85, axis=1)*W[263]
    return(result)

@tf.function
def node1448(X):
    result = B[1448]
    return(result)

@tf.function
def node1449(X):
    result = B[1449] + node1262(X)*W[27] + node1536(X)*W[357]
    return(result)

@tf.function
def node1450(X):
    result = B[1450]
    return(result)

@tf.function
def node1451(X):
    result = B[1451] + tf.gather(X, 514, axis=1)*W[52] + node1718(X)*W[315]
    return(result)

@tf.function
def node1452(X):
    result = B[1452] + tf.gather(X, 483, axis=1)*W[371]
    return(result)

@tf.function
def node1453(X):
    result = B[1453] + node1740(X)*W[501]
    return(result)

@tf.function
def node1454(X):
    result = B[1454] + node1273(X)*W[860]
    return(result)

@tf.function
def node1455(X):
    result = B[1455]
    return(result)

@tf.function
def node1456(X):
    result = B[1456] + tf.gather(X, 104, axis=1)*W[493] + node1058(X)*W[286] + node1120(X)*W[574]
    return(result)

@tf.function
def node1457(X):
    result = B[1457]
    return(result)

@tf.function
def node1458(X):
    result = B[1458] + tf.gather(X, 167, axis=1)*W[641] + tf.gather(X, 691, axis=1)*W[276] + node1531(X)*W[966]
    return(result)

@tf.function
def node1459(X):
    result = B[1459]
    return(result)

@tf.function
def node1460(X):
    result = B[1460] + tf.gather(X, 175, axis=1)*W[238] + tf.gather(X, 573, axis=1)*W[587]
    return(result)

@tf.function
def node1461(X):
    result = B[1461] + tf.gather(X, 206, axis=1)*W[838] + node1549(X)*W[252]
    return(result)

@tf.function
def node1462(X):
    result = B[1462]
    return(result)

@tf.function
def node1463(X):
    result = B[1463] + tf.gather(X, 358, axis=1)*W[712]
    return(result)

@tf.function
def node1464(X):
    result = B[1464]
    return(result)

@tf.function
def node1465(X):
    result = B[1465]
    return(result)

@tf.function
def node1466(X):
    result = B[1466] + tf.gather(X, 606, axis=1)*W[941]
    return(result)

@tf.function
def node1467(X):
    result = B[1467]
    return(result)

@tf.function
def node1468(X):
    result = B[1468]
    return(result)

@tf.function
def node1469(X):
    result = B[1469] + node1188(X)*W[678]
    return(result)

@tf.function
def node1470(X):
    result = B[1470] + tf.gather(X, 573, axis=1)*W[302]
    return(result)

@tf.function
def node1471(X):
    result = B[1471] + node906(X)*W[703] + node1392(X)*W[975] + node1607(X)*W[577]
    return(result)

@tf.function
def node1472(X):
    result = B[1472] + tf.gather(X, 778, axis=1)*W[343]
    return(result)

@tf.function
def node1473(X):
    result = B[1473] + tf.gather(X, 410, axis=1)*W[740] + node880(X)*W[132] + node1279(X)*W[942]
    return(result)

@tf.function
def node1474(X):
    result = B[1474] + tf.gather(X, 186, axis=1)*W[592]
    return(result)

@tf.function
def node1475(X):
    result = B[1475] + tf.gather(X, 542, axis=1)*W[657]
    return(result)

@tf.function
def node1476(X):
    result = B[1476]
    return(result)

@tf.function
def node1477(X):
    result = B[1477]
    return(result)

@tf.function
def node1478(X):
    result = B[1478] + tf.gather(X, 74, axis=1)*W[406] + tf.gather(X, 307, axis=1)*W[789]
    return(result)

@tf.function
def node1479(X):
    result = B[1479] + node851(X)*W[614]
    return(result)

@tf.function
def node1480(X):
    result = B[1480]
    return(result)

@tf.function
def node1481(X):
    result = B[1481] + node1158(X)*W[247]
    return(result)

@tf.function
def node1482(X):
    result = B[1482] + tf.gather(X, 499, axis=1)*W[974] + tf.gather(X, 573, axis=1)*W[822] + node1129(X)*W[840]
    return(result)

@tf.function
def node1483(X):
    result = B[1483] + node1020(X)*W[954]
    return(result)

@tf.function
def node1484(X):
    result = B[1484] + node1415(X)*W[452]
    return(result)

@tf.function
def node1485(X):
    result = B[1485] + node1438(X)*W[370]
    return(result)

@tf.function
def node1486(X):
    result = B[1486]
    return(result)

@tf.function
def node1487(X):
    result = B[1487]
    return(result)

@tf.function
def node1488(X):
    result = B[1488] + tf.gather(X, 520, axis=1)*W[381] + node1399(X)*W[34]
    return(result)

@tf.function
def node1489(X):
    result = B[1489] + node1151(X)*W[820]
    return(result)

@tf.function
def node1490(X):
    result = B[1490] + node1202(X)*W[938]
    return(result)

@tf.function
def node1491(X):
    result = B[1491] + tf.gather(X, 658, axis=1)*W[799] + tf.gather(X, 748, axis=1)*W[221]
    return(result)

@tf.function
def node1492(X):
    result = B[1492]
    return(result)

@tf.function
def node1493(X):
    result = B[1493]
    return(result)

@tf.function
def node1494(X):
    result = B[1494]
    return(result)

@tf.function
def node1495(X):
    result = B[1495]
    return(result)

@tf.function
def node1496(X):
    result = B[1496]
    return(result)

@tf.function
def node1497(X):
    result = B[1497]
    return(result)

@tf.function
def node1498(X):
    result = B[1498] + tf.gather(X, 694, axis=1)*W[426]
    return(result)

@tf.function
def node1499(X):
    result = B[1499] + node1127(X)*W[102]
    return(result)

@tf.function
def node1500(X):
    result = B[1500] + tf.gather(X, 274, axis=1)*W[324]
    return(result)

@tf.function
def node1501(X):
    result = B[1501] + tf.gather(X, 276, axis=1)*W[285]
    return(result)

@tf.function
def node1502(X):
    result = B[1502]
    return(result)

@tf.function
def node1503(X):
    result = B[1503] + tf.gather(X, 384, axis=1)*W[894]
    return(result)

@tf.function
def node1504(X):
    result = B[1504]
    return(result)

@tf.function
def node1505(X):
    result = B[1505] + tf.gather(X, 13, axis=1)*W[907] + tf.gather(X, 89, axis=1)*W[353]
    return(result)

@tf.function
def node1506(X):
    result = B[1506] + tf.gather(X, 62, axis=1)*W[842] + tf.gather(X, 70, axis=1)*W[448]
    return(result)

@tf.function
def node1507(X):
    result = B[1507] + tf.gather(X, 102, axis=1)*W[559] + tf.gather(X, 570, axis=1)*W[430] + node1244(X)*W[460] + node1247(X)*W[373]
    return(result)

@tf.function
def node1508(X):
    result = B[1508]
    return(result)

@tf.function
def node1509(X):
    result = B[1509] + tf.gather(X, 355, axis=1)*W[200] + tf.gather(X, 591, axis=1)*W[462]
    return(result)

@tf.function
def node1510(X):
    result = B[1510]
    return(result)

@tf.function
def node1511(X):
    result = B[1511]
    return(result)

@tf.function
def node1512(X):
    result = B[1512]
    return(result)

@tf.function
def node1513(X):
    result = B[1513] + tf.gather(X, 733, axis=1)*W[794] + node1535(X)*W[690]
    return(result)

@tf.function
def node1514(X):
    result = B[1514] + node1674(X)*W[69]
    return(result)

@tf.function
def node1515(X):
    result = B[1515] + node1388(X)*W[958]
    return(result)

@tf.function
def node1516(X):
    result = B[1516]
    return(result)

@tf.function
def node1517(X):
    result = B[1517]
    return(result)

@tf.function
def node1518(X):
    result = B[1518] + tf.gather(X, 357, axis=1)*W[104]
    return(result)

@tf.function
def node1519(X):
    result = B[1519] + tf.gather(X, 287, axis=1)*W[62] + tf.gather(X, 430, axis=1)*W[865] + node1244(X)*W[520]
    return(result)

@tf.function
def node1520(X):
    result = B[1520] + tf.gather(X, 715, axis=1)*W[437] + node1183(X)*W[835]
    return(result)

@tf.function
def node1521(X):
    result = B[1521]
    return(result)

@tf.function
def node1522(X):
    result = B[1522] + node1768(X)*W[486]
    return(result)

@tf.function
def node1523(X):
    result = B[1523] + node1137(X)*W[454]
    return(result)

@tf.function
def node1524(X):
    result = B[1524] + tf.gather(X, 728, axis=1)*W[832] + node1004(X)*W[646]
    return(result)

@tf.function
def node1525(X):
    result = B[1525] + node994(X)*W[54]
    return(result)

@tf.function
def node1526(X):
    result = B[1526] + node1624(X)*W[816] + node1659(X)*W[560]
    return(result)

@tf.function
def node1527(X):
    result = B[1527]
    return(result)

@tf.function
def node1528(X):
    result = B[1528] + node1390(X)*W[887]
    return(result)

@tf.function
def node1529(X):
    result = B[1529] + node803(X)*W[825]
    return(result)

@tf.function
def node1530(X):
    result = B[1530] + node1026(X)*W[671] + node1424(X)*W[859]
    return(result)

@tf.function
def node1531(X):
    result = B[1531] + node1165(X)*W[716] + node1508(X)*W[408]
    return(result)

@tf.function
def node1532(X):
    result = B[1532] + tf.gather(X, 360, axis=1)*W[515] + tf.gather(X, 416, axis=1)*W[167] + tf.gather(X, 520, axis=1)*W[936]
    return(result)

@tf.function
def node1533(X):
    result = B[1533]
    return(result)

@tf.function
def node1534(X):
    result = B[1534] + tf.gather(X, 268, axis=1)*W[481] + node958(X)*W[284]
    return(result)

@tf.function
def node1535(X):
    result = B[1535]
    return(result)

@tf.function
def node1536(X):
    result = B[1536] + node1790(X)*W[546]
    return(result)

@tf.function
def node1537(X):
    result = B[1537] + tf.gather(X, 297, axis=1)*W[891]
    return(result)

@tf.function
def node1538(X):
    result = B[1538] + tf.gather(X, 49, axis=1)*W[736] + node1677(X)*W[195]
    return(result)

@tf.function
def node1539(X):
    result = B[1539] + node1142(X)*W[435]
    return(result)

@tf.function
def node1540(X):
    result = B[1540] + tf.gather(X, 239, axis=1)*W[754] + node1126(X)*W[507]
    return(result)

@tf.function
def node1541(X):
    result = B[1541]
    return(result)

@tf.function
def node1542(X):
    result = B[1542] + node1113(X)*W[436] + node1482(X)*W[843]
    return(result)

@tf.function
def node1543(X):
    result = B[1543] + tf.gather(X, 432, axis=1)*W[311] + tf.gather(X, 440, axis=1)*W[190] + node1277(X)*W[852]
    return(result)

@tf.function
def node1544(X):
    result = B[1544] + tf.gather(X, 466, axis=1)*W[836] + tf.gather(X, 762, axis=1)*W[82] + node1367(X)*W[127]
    return(result)

@tf.function
def node1545(X):
    result = B[1545]
    return(result)

@tf.function
def node1546(X):
    result = B[1546] + node1320(X)*W[292] + node1543(X)*W[923]
    return(result)

@tf.function
def node1547(X):
    result = B[1547] + tf.gather(X, 463, axis=1)*W[219] + node1058(X)*W[547]
    return(result)

@tf.function
def node1548(X):
    result = B[1548] + tf.gather(X, 24, axis=1)*W[110] + node1132(X)*W[365] + node1249(X)*W[428] + node1749(X)*W[178]
    return(result)

@tf.function
def node1549(X):
    result = B[1549]
    return(result)

@tf.function
def node1550(X):
    result = B[1550] + node1069(X)*W[403] + node1163(X)*W[770] + node1265(X)*W[366]
    return(result)

@tf.function
def node1551(X):
    result = B[1551]
    return(result)

@tf.function
def node1552(X):
    result = B[1552] + tf.gather(X, 442, axis=1)*W[595] + tf.gather(X, 514, axis=1)*W[924] + node1415(X)*W[489] + node1578(X)*W[119]
    return(result)

@tf.function
def node1553(X):
    result = B[1553] + node832(X)*W[851] + node1238(X)*W[137]
    return(result)

@tf.function
def node1554(X):
    result = B[1554] + tf.gather(X, 658, axis=1)*W[291] + tf.gather(X, 755, axis=1)*W[839] + node1614(X)*W[575]
    return(result)

@tf.function
def node1555(X):
    result = B[1555] + tf.gather(X, 417, axis=1)*W[429] + tf.gather(X, 661, axis=1)*W[197] + node956(X)*W[993]
    return(result)

@tf.function
def node1556(X):
    result = B[1556]
    return(result)

@tf.function
def node1557(X):
    result = B[1557]
    return(result)

@tf.function
def node1558(X):
    result = B[1558]
    return(result)

@tf.function
def node1559(X):
    result = B[1559]
    return(result)

@tf.function
def node1560(X):
    result = B[1560] + tf.gather(X, 477, axis=1)*W[297] + node1426(X)*W[83] + node1439(X)*W[931]
    return(result)

@tf.function
def node1561(X):
    result = B[1561] + tf.gather(X, 622, axis=1)*W[130]
    return(result)

@tf.function
def node1562(X):
    result = B[1562] + node1660(X)*W[666]
    return(result)

@tf.function
def node1563(X):
    result = B[1563] + tf.gather(X, 11, axis=1)*W[651] + node1237(X)*W[554]
    return(result)

@tf.function
def node1564(X):
    result = B[1564] + node1534(X)*W[253]
    return(result)

@tf.function
def node1565(X):
    result = B[1565] + tf.gather(X, 103, axis=1)*W[80] + tf.gather(X, 175, axis=1)*W[688] + node1004(X)*W[693]
    return(result)

@tf.function
def node1566(X):
    result = B[1566] + node1359(X)*W[904]
    return(result)

@tf.function
def node1567(X):
    result = B[1567] + node1076(X)*W[465]
    return(result)

@tf.function
def node1568(X):
    result = B[1568]
    return(result)

@tf.function
def node1569(X):
    result = B[1569]
    return(result)

@tf.function
def node1570(X):
    result = B[1570]
    return(result)

@tf.function
def node1571(X):
    result = B[1571] + tf.gather(X, 666, axis=1)*W[521]
    return(result)

@tf.function
def node1572(X):
    result = B[1572] + tf.gather(X, 543, axis=1)*W[526] + node1454(X)*W[767]
    return(result)

@tf.function
def node1573(X):
    result = B[1573]
    return(result)

@tf.function
def node1574(X):
    result = B[1574] + node1378(X)*W[438]
    return(result)

@tf.function
def node1575(X):
    result = B[1575]
    return(result)

@tf.function
def node1576(X):
    result = B[1576] + node1243(X)*W[605]
    return(result)

@tf.function
def node1577(X):
    result = B[1577]
    return(result)

@tf.function
def node1578(X):
    result = B[1578]
    return(result)

@tf.function
def node1579(X):
    result = B[1579] + node1538(X)*W[355]
    return(result)

@tf.function
def node1580(X):
    result = B[1580] + tf.gather(X, 744, axis=1)*W[659] + node1336(X)*W[757]
    return(result)

@tf.function
def node1581(X):
    result = B[1581] + node1553(X)*W[495]
    return(result)

@tf.function
def node1582(X):
    result = B[1582] + tf.gather(X, 504, axis=1)*W[682] + tf.gather(X, 621, axis=1)*W[847] + tf.gather(X, 624, axis=1)*W[467] + node1020(X)*W[898]
    return(result)

@tf.function
def node1583(X):
    result = B[1583]
    return(result)

@tf.function
def node1584(X):
    result = B[1584] + node1768(X)*W[910]
    return(result)

@tf.function
def node1585(X):
    result = B[1585] + node1447(X)*W[782]
    return(result)

@tf.function
def node1586(X):
    result = B[1586]
    return(result)

@tf.function
def node1587(X):
    result = B[1587] + tf.gather(X, 14, axis=1)*W[265] + tf.gather(X, 267, axis=1)*W[59] + node1295(X)*W[203]
    return(result)

@tf.function
def node1588(X):
    result = B[1588] + tf.gather(X, 184, axis=1)*W[579]
    return(result)

@tf.function
def node1589(X):
    result = B[1589] + node1215(X)*W[684] + node1320(X)*W[670]
    return(result)

@tf.function
def node1590(X):
    result = B[1590]
    return(result)

@tf.function
def node1591(X):
    result = B[1591]
    return(result)

@tf.function
def node1592(X):
    result = B[1592] + node837(X)*W[543]
    return(result)

@tf.function
def node1593(X):
    result = B[1593]
    return(result)

@tf.function
def node1594(X):
    result = B[1594] + node1032(X)*W[616] + node1551(X)*W[612]
    return(result)

@tf.function
def node1595(X):
    result = B[1595] + node1422(X)*W[417]
    return(result)

@tf.function
def node1596(X):
    result = B[1596]
    return(result)

@tf.function
def node1597(X):
    result = B[1597] + node1454(X)*W[890]
    return(result)

@tf.function
def node1598(X):
    result = B[1598]
    return(result)

@tf.function
def node1599(X):
    result = B[1599] + tf.gather(X, 590, axis=1)*W[434] + tf.gather(X, 703, axis=1)*W[161] + node1011(X)*W[356]
    return(result)

@tf.function
def node1600(X):
    result = B[1600] + tf.gather(X, 269, axis=1)*W[323]
    return(result)

@tf.function
def node1601(X):
    result = B[1601]
    return(result)

@tf.function
def node1602(X):
    result = B[1602] + node1040(X)*W[250]
    return(result)

@tf.function
def node1603(X):
    result = B[1603]
    return(result)

@tf.function
def node1604(X):
    result = B[1604] + tf.gather(X, 2, axis=1)*W[819] + node1530(X)*W[147] + node1601(X)*W[229] + node1661(X)*W[352]
    return(result)

@tf.function
def node1605(X):
    result = B[1605] + tf.gather(X, 236, axis=1)*W[540] + tf.gather(X, 489, axis=1)*W[578] + tf.gather(X, 560, axis=1)*W[812]
    return(result)

@tf.function
def node1606(X):
    result = B[1606] + tf.gather(X, 514, axis=1)*W[960] + node913(X)*W[208]
    return(result)

@tf.function
def node1607(X):
    result = B[1607] + tf.gather(X, 65, axis=1)*W[648]
    return(result)

@tf.function
def node1608(X):
    result = B[1608]
    return(result)

@tf.function
def node1609(X):
    result = B[1609]
    return(result)

@tf.function
def node1610(X):
    result = B[1610] + node1273(X)*W[990]
    return(result)

@tf.function
def node1611(X):
    result = B[1611] + tf.gather(X, 361, axis=1)*W[722] + tf.gather(X, 435, axis=1)*W[765]
    return(result)

@tf.function
def node1612(X):
    result = B[1612] + tf.gather(X, 24, axis=1)*W[818] + node1271(X)*W[594]
    return(result)

@tf.function
def node1613(X):
    result = B[1613] + node922(X)*W[424]
    return(result)

@tf.function
def node1614(X):
    result = B[1614] + tf.gather(X, 254, axis=1)*W[885] + node1659(X)*W[40]
    return(result)

@tf.function
def node1615(X):
    result = B[1615]
    return(result)

@tf.function
def node1616(X):
    result = B[1616] + tf.gather(X, 311, axis=1)*W[260] + node937(X)*W[850]
    return(result)

@tf.function
def node1617(X):
    result = B[1617] + tf.gather(X, 14, axis=1)*W[933]
    return(result)

@tf.function
def node1618(X):
    result = B[1618]
    return(result)

@tf.function
def node1619(X):
    result = B[1619]
    return(result)

@tf.function
def node1620(X):
    result = B[1620] + tf.gather(X, 34, axis=1)*W[245] + node1721(X)*W[85]
    return(result)

@tf.function
def node1621(X):
    result = B[1621]
    return(result)

@tf.function
def node1622(X):
    result = B[1622] + node1038(X)*W[561] + node1049(X)*W[368]
    return(result)

@tf.function
def node1623(X):
    result = B[1623]
    return(result)

@tf.function
def node1624(X):
    result = B[1624] + node1204(X)*W[425]
    return(result)

@tf.function
def node1625(X):
    result = B[1625]
    return(result)

@tf.function
def node1626(X):
    result = B[1626] + tf.gather(X, 717, axis=1)*W[977]
    return(result)

@tf.function
def node1627(X):
    result = B[1627]
    return(result)

@tf.function
def node1628(X):
    result = B[1628]
    return(result)

@tf.function
def node1629(X):
    result = B[1629]
    return(result)

@tf.function
def node1630(X):
    result = B[1630]
    return(result)

@tf.function
def node1631(X):
    result = B[1631]
    return(result)

@tf.function
def node1632(X):
    result = B[1632] + tf.gather(X, 727, axis=1)*W[881] + node1561(X)*W[882]
    return(result)

@tf.function
def node1633(X):
    result = B[1633] + node1148(X)*W[183] + node1438(X)*W[875]
    return(result)

@tf.function
def node1634(X):
    result = B[1634]
    return(result)

@tf.function
def node1635(X):
    result = B[1635]
    return(result)

@tf.function
def node1636(X):
    result = B[1636]
    return(result)

@tf.function
def node1637(X):
    result = B[1637] + node1257(X)*W[509] + node1732(X)*W[236]
    return(result)

@tf.function
def node1638(X):
    result = B[1638]
    return(result)

@tf.function
def node1639(X):
    result = B[1639]
    return(result)

@tf.function
def node1640(X):
    result = B[1640] + tf.gather(X, 41, axis=1)*W[597]
    return(result)

@tf.function
def node1641(X):
    result = B[1641] + node978(X)*W[191]
    return(result)

@tf.function
def node1642(X):
    result = B[1642] + node807(X)*W[934]
    return(result)

@tf.function
def node1643(X):
    result = B[1643] + tf.gather(X, 173, axis=1)*W[878] + node1213(X)*W[405] + node1446(X)*W[981]
    return(result)

@tf.function
def node1644(X):
    result = B[1644]
    return(result)

@tf.function
def node1645(X):
    result = B[1645]
    return(result)

@tf.function
def node1646(X):
    result = B[1646] + node1316(X)*W[474]
    return(result)

@tf.function
def node1647(X):
    result = B[1647] + node1064(X)*W[378]
    return(result)

@tf.function
def node1648(X):
    result = B[1648] + node898(X)*W[506]
    return(result)

@tf.function
def node1649(X):
    result = B[1649] + tf.gather(X, 348, axis=1)*W[709]
    return(result)

@tf.function
def node1650(X):
    result = B[1650] + tf.gather(X, 104, axis=1)*W[22]
    return(result)

@tf.function
def node1651(X):
    result = B[1651]
    return(result)

@tf.function
def node1652(X):
    result = B[1652] + node1425(X)*W[902]
    return(result)

@tf.function
def node1653(X):
    result = B[1653] + node855(X)*W[306]
    return(result)

@tf.function
def node1654(X):
    result = B[1654] + node1647(X)*W[257]
    return(result)

@tf.function
def node1655(X):
    result = B[1655]
    return(result)

@tf.function
def node1656(X):
    result = B[1656] + node963(X)*W[35]
    return(result)

@tf.function
def node1657(X):
    result = B[1657] + tf.gather(X, 35, axis=1)*W[999]
    return(result)

@tf.function
def node1658(X):
    result = B[1658] + node1370(X)*W[655]
    return(result)

@tf.function
def node1659(X):
    result = B[1659]
    return(result)

@tf.function
def node1660(X):
    result = B[1660]
    return(result)

@tf.function
def node1661(X):
    result = B[1661]
    return(result)

@tf.function
def node1662(X):
    result = B[1662]
    return(result)

@tf.function
def node1663(X):
    result = B[1663]
    return(result)

@tf.function
def node1664(X):
    result = B[1664] + tf.gather(X, 258, axis=1)*W[498]
    return(result)

@tf.function
def node1665(X):
    result = B[1665]
    return(result)

@tf.function
def node1666(X):
    result = B[1666] + tf.gather(X, 309, axis=1)*W[158] + node1212(X)*W[274] + node1497(X)*W[439]
    return(result)

@tf.function
def node1667(X):
    result = B[1667] + node806(X)*W[957]
    return(result)

@tf.function
def node1668(X):
    result = B[1668] + node1577(X)*W[281]
    return(result)

@tf.function
def node1669(X):
    result = B[1669]
    return(result)

@tf.function
def node1670(X):
    result = B[1670] + node1245(X)*W[463]
    return(result)

@tf.function
def node1671(X):
    result = B[1671] + tf.gather(X, 529, axis=1)*W[653]
    return(result)

@tf.function
def node1672(X):
    result = B[1672]
    return(result)

@tf.function
def node1673(X):
    result = B[1673] + node1106(X)*W[801]
    return(result)

@tf.function
def node1674(X):
    result = B[1674] + tf.gather(X, 650, axis=1)*W[714]
    return(result)

@tf.function
def node1675(X):
    result = B[1675]
    return(result)

@tf.function
def node1676(X):
    result = B[1676] + tf.gather(X, 469, axis=1)*W[727]
    return(result)

@tf.function
def node1677(X):
    result = B[1677] + node1161(X)*W[335]
    return(result)

@tf.function
def node1678(X):
    result = B[1678] + tf.gather(X, 212, axis=1)*W[449]
    return(result)

@tf.function
def node1679(X):
    result = B[1679] + tf.gather(X, 465, axis=1)*W[980] + tf.gather(X, 532, axis=1)*W[308] + node1324(X)*W[410]
    return(result)

@tf.function
def node1680(X):
    result = B[1680]
    return(result)

@tf.function
def node1681(X):
    result = B[1681]
    return(result)

@tf.function
def node1682(X):
    result = B[1682]
    return(result)

@tf.function
def node1683(X):
    result = B[1683]
    return(result)

@tf.function
def node1684(X):
    result = B[1684]
    return(result)

@tf.function
def node1685(X):
    result = B[1685] + tf.gather(X, 327, axis=1)*W[453] + tf.gather(X, 558, axis=1)*W[220]
    return(result)

@tf.function
def node1686(X):
    result = B[1686]
    return(result)

@tf.function
def node1687(X):
    result = B[1687] + tf.gather(X, 521, axis=1)*W[67]
    return(result)

@tf.function
def node1688(X):
    result = B[1688] + tf.gather(X, 609, axis=1)*W[216] + node1286(X)*W[383]
    return(result)

@tf.function
def node1689(X):
    result = B[1689]
    return(result)

@tf.function
def node1690(X):
    result = B[1690] + node1392(X)*W[143]
    return(result)

@tf.function
def node1691(X):
    result = B[1691] + tf.gather(X, 515, axis=1)*W[549]
    return(result)

@tf.function
def node1692(X):
    result = B[1692] + node1532(X)*W[49] + node1604(X)*W[855]
    return(result)

@tf.function
def node1693(X):
    result = B[1693] + tf.gather(X, 352, axis=1)*W[919]
    return(result)

@tf.function
def node1694(X):
    result = B[1694] + node1335(X)*W[327] + node1702(X)*W[194]
    return(result)

@tf.function
def node1695(X):
    result = B[1695] + node1466(X)*W[766]
    return(result)

@tf.function
def node1696(X):
    result = B[1696]
    return(result)

@tf.function
def node1697(X):
    result = B[1697]
    return(result)

@tf.function
def node1698(X):
    result = B[1698] + node1548(X)*W[541]
    return(result)

@tf.function
def node1699(X):
    result = B[1699]
    return(result)

@tf.function
def node1700(X):
    result = B[1700] + tf.gather(X, 12, axis=1)*W[415] + node1235(X)*W[619]
    return(result)

@tf.function
def node1701(X):
    result = B[1701]
    return(result)

@tf.function
def node1702(X):
    result = B[1702] + node1689(X)*W[482]
    return(result)

@tf.function
def node1703(X):
    result = B[1703]
    return(result)

@tf.function
def node1704(X):
    result = B[1704] + node1550(X)*W[139]
    return(result)

@tf.function
def node1705(X):
    result = B[1705] + tf.gather(X, 535, axis=1)*W[146] + tf.gather(X, 710, axis=1)*W[445] + node1683(X)*W[157]
    return(result)

@tf.function
def node1706(X):
    result = B[1706]
    return(result)

@tf.function
def node1707(X):
    result = B[1707]
    return(result)

@tf.function
def node1708(X):
    result = B[1708]
    return(result)

@tf.function
def node1709(X):
    result = B[1709] + tf.gather(X, 598, axis=1)*W[596] + node1224(X)*W[407]
    return(result)

@tf.function
def node1710(X):
    result = B[1710]
    return(result)

@tf.function
def node1711(X):
    result = B[1711]
    return(result)

@tf.function
def node1712(X):
    result = B[1712] + node1358(X)*W[103]
    return(result)

@tf.function
def node1713(X):
    result = B[1713] + tf.gather(X, 90, axis=1)*W[57] + tf.gather(X, 446, axis=1)*W[996]
    return(result)

@tf.function
def node1714(X):
    result = B[1714] + tf.gather(X, 64, axis=1)*W[487]
    return(result)

@tf.function
def node1715(X):
    result = B[1715] + node1400(X)*W[517]
    return(result)

@tf.function
def node1716(X):
    result = B[1716]
    return(result)

@tf.function
def node1717(X):
    result = B[1717] + node1323(X)*W[785]
    return(result)

@tf.function
def node1718(X):
    result = B[1718] + node1362(X)*W[65]
    return(result)

@tf.function
def node1719(X):
    result = B[1719]
    return(result)

@tf.function
def node1720(X):
    result = B[1720] + tf.gather(X, 655, axis=1)*W[621]
    return(result)

@tf.function
def node1721(X):
    result = B[1721] + tf.gather(X, 536, axis=1)*W[696]
    return(result)

@tf.function
def node1722(X):
    result = B[1722]
    return(result)

@tf.function
def node1723(X):
    result = B[1723] + node1667(X)*W[75]
    return(result)

@tf.function
def node1724(X):
    result = B[1724]
    return(result)

@tf.function
def node1725(X):
    result = B[1725]
    return(result)

@tf.function
def node1726(X):
    result = B[1726] + tf.gather(X, 103, axis=1)*W[656]
    return(result)

@tf.function
def node1727(X):
    result = B[1727] + node1460(X)*W[516]
    return(result)

@tf.function
def node1728(X):
    result = B[1728] + tf.gather(X, 627, axis=1)*W[861]
    return(result)

@tf.function
def node1729(X):
    result = B[1729]
    return(result)

@tf.function
def node1730(X):
    result = B[1730] + node846(X)*W[834]
    return(result)

@tf.function
def node1731(X):
    result = B[1731] + tf.gather(X, 206, axis=1)*W[96] + node1457(X)*W[823] + node1565(X)*W[837]
    return(result)

@tf.function
def node1732(X):
    result = B[1732] + tf.gather(X, 146, axis=1)*W[264]
    return(result)

@tf.function
def node1733(X):
    result = B[1733] + tf.gather(X, 284, axis=1)*W[943] + tf.gather(X, 430, axis=1)*W[982] + tf.gather(X, 534, axis=1)*W[743] + node1240(X)*W[350]
    return(result)

@tf.function
def node1734(X):
    result = B[1734] + tf.gather(X, 175, axis=1)*W[165] + node1739(X)*W[976]
    return(result)

@tf.function
def node1735(X):
    result = B[1735] + node1646(X)*W[333]
    return(result)

@tf.function
def node1736(X):
    result = B[1736] + tf.gather(X, 700, axis=1)*W[781]
    return(result)

@tf.function
def node1737(X):
    result = B[1737] + tf.gather(X, 517, axis=1)*W[673] + tf.gather(X, 590, axis=1)*W[91]
    return(result)

@tf.function
def node1738(X):
    result = B[1738] + node842(X)*W[802]
    return(result)

@tf.function
def node1739(X):
    result = B[1739] + tf.gather(X, 11, axis=1)*W[750] + tf.gather(X, 132, axis=1)*W[100]
    return(result)

@tf.function
def node1740(X):
    result = B[1740]
    return(result)

@tf.function
def node1741(X):
    result = B[1741] + node1117(X)*W[830] + node1207(X)*W[36]
    return(result)

@tf.function
def node1742(X):
    result = B[1742]
    return(result)

@tf.function
def node1743(X):
    result = B[1743]
    return(result)

@tf.function
def node1744(X):
    result = B[1744] + tf.gather(X, 779, axis=1)*W[786]
    return(result)

@tf.function
def node1745(X):
    result = B[1745] + node1041(X)*W[420] + node1143(X)*W[442]
    return(result)

@tf.function
def node1746(X):
    result = B[1746] + tf.gather(X, 622, axis=1)*W[354] + node967(X)*W[741]
    return(result)

@tf.function
def node1747(X):
    result = B[1747] + node1366(X)*W[74] + node1378(X)*W[470]
    return(result)

@tf.function
def node1748(X):
    result = B[1748]
    return(result)

@tf.function
def node1749(X):
    result = B[1749]
    return(result)

@tf.function
def node1750(X):
    result = B[1750] + tf.gather(X, 93, axis=1)*W[635] + node1596(X)*W[920]
    return(result)

@tf.function
def node1751(X):
    result = B[1751]
    return(result)

@tf.function
def node1752(X):
    result = B[1752] + tf.gather(X, 146, axis=1)*W[382] + node1448(X)*W[994]
    return(result)

@tf.function
def node1753(X):
    result = B[1753] + node1350(X)*W[715]
    return(result)

@tf.function
def node1754(X):
    result = B[1754]
    return(result)

@tf.function
def node1755(X):
    result = B[1755]
    return(result)

@tf.function
def node1756(X):
    result = B[1756]
    return(result)

@tf.function
def node1757(X):
    result = B[1757] + node1152(X)*W[246]
    return(result)

@tf.function
def node1758(X):
    result = B[1758] + tf.gather(X, 484, axis=1)*W[953]
    return(result)

@tf.function
def node1759(X):
    result = B[1759] + tf.gather(X, 11, axis=1)*W[564] + node968(X)*W[156]
    return(result)

@tf.function
def node1760(X):
    result = B[1760] + node1651(X)*W[272]
    return(result)

@tf.function
def node1761(X):
    result = B[1761] + node1099(X)*W[86]
    return(result)

@tf.function
def node1762(X):
    result = B[1762] + node815(X)*W[364] + node860(X)*W[25] + node1640(X)*W[204]
    return(result)

@tf.function
def node1763(X):
    result = B[1763]
    return(result)

@tf.function
def node1764(X):
    result = B[1764] + tf.gather(X, 403, axis=1)*W[730] + node1102(X)*W[548]
    return(result)

@tf.function
def node1765(X):
    result = B[1765] + tf.gather(X, 33, axis=1)*W[330] + tf.gather(X, 514, axis=1)*W[73]
    return(result)

@tf.function
def node1766(X):
    result = B[1766] + node808(X)*W[301]
    return(result)

@tf.function
def node1767(X):
    result = B[1767] + node1060(X)*W[984] + node1299(X)*W[821]
    return(result)

@tf.function
def node1768(X):
    result = B[1768] + node1568(X)*W[748]
    return(result)

@tf.function
def node1769(X):
    result = B[1769] + node1002(X)*W[249]
    return(result)

@tf.function
def node1770(X):
    result = B[1770] + node953(X)*W[824]
    return(result)

@tf.function
def node1771(X):
    result = B[1771] + node1318(X)*W[749] + node1340(X)*W[639]
    return(result)

@tf.function
def node1772(X):
    result = B[1772] + node1068(X)*W[92]
    return(result)

@tf.function
def node1773(X):
    result = B[1773]
    return(result)

@tf.function
def node1774(X):
    result = B[1774] + node1512(X)*W[17]
    return(result)

@tf.function
def node1775(X):
    result = B[1775] + node1036(X)*W[279]
    return(result)

@tf.function
def node1776(X):
    result = B[1776]
    return(result)

@tf.function
def node1777(X):
    result = B[1777] + tf.gather(X, 260, axis=1)*W[634]
    return(result)

@tf.function
def node1778(X):
    result = B[1778]
    return(result)

@tf.function
def node1779(X):
    result = B[1779] + tf.gather(X, 21, axis=1)*W[761]
    return(result)

@tf.function
def node1780(X):
    result = B[1780] + tf.gather(X, 11, axis=1)*W[12] + node971(X)*W[689]
    return(result)

@tf.function
def node1781(X):
    result = B[1781]
    return(result)

@tf.function
def node1782(X):
    result = B[1782] + tf.gather(X, 110, axis=1)*W[970]
    return(result)

@tf.function
def node1783(X):
    result = B[1783] + tf.gather(X, 396, axis=1)*W[685] + tf.gather(X, 437, axis=1)*W[433] + node872(X)*W[557]
    return(result)

@tf.function
def node1784(X):
    result = B[1784]
    return(result)

@tf.function
def node1785(X):
    result = B[1785] + tf.gather(X, 316, axis=1)*W[805]
    return(result)

@tf.function
def node1786(X):
    result = B[1786] + node1355(X)*W[570]
    return(result)

@tf.function
def node1787(X):
    result = B[1787]
    return(result)

@tf.function
def node1788(X):
    result = B[1788] + node823(X)*W[273] + node1435(X)*W[888]
    return(result)

@tf.function
def node1789(X):
    result = B[1789]
    return(result)

@tf.function
def node1790(X):
    result = B[1790] + tf.gather(X, 361, axis=1)*W[258]
    return(result)

@tf.function
def node1791(X):
    result = B[1791]
    return(result)

@tf.function
def node1792(X):
    result = B[1792] + tf.gather(X, 306, axis=1)*W[133] + tf.gather(X, 571, axis=1)*W[711]
    return(result)

@tf.function
def node1793(X):
    result = B[1793] + node1096(X)*W[783]
    return(result)

@tf.function
def Hypothesis(X):
    out0 = B[784] + tf.gather(X, 294, axis=1)*W[0]
    out1 = B[785] + tf.gather(X, 624, axis=1)*W[1]
    out2 = B[786] + tf.gather(X, 725, axis=1)*W[2] + node1259(X)*W[604]
    out3 = B[787] + tf.gather(X, 111, axis=1)*W[3] + node1576(X)*W[879] + node1771(X)*W[446]
    out4 = B[788] + tf.gather(X, 507, axis=1)*W[4] + tf.gather(X, 752, axis=1)*W[37] + node1793(X)*W[610]
    out5 = B[789] + tf.gather(X, 452, axis=1)*W[5] + tf.gather(X, 551, axis=1)*W[142]
    out6 = B[790] + tf.gather(X, 95, axis=1)*W[6] + node1077(X)*W[173] + node1740(X)*W[896]
    out7 = B[791] + tf.gather(X, 262, axis=1)*W[7] + node1759(X)*W[348]
    out8 = B[792] + tf.gather(X, 98, axis=1)*W[53] + tf.gather(X, 199, axis=1)*W[8]
    out9 = B[793] + tf.gather(X, 230, axis=1)*W[9]
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

print('Learning finished')
print('Accuracy = {:.4f}'.format(Accuracy(x_test, y_test)))
