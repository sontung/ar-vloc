import torch
import torch.nn
import sys
import numpy as np
import pnp.build.pnp_python_binding


d1 = [[1.0213794597535921, -1.107638252964459, 4.007113416577096], [-3.3422326697524403, -0.2090180682076106, -0.5035807697826078], [-1.8604552460208428, -2.8488779194813203, -0.5341833005986398], [-4.684759976434113, -3.2729500127735895, -2.773934823780767], [5.641738486479504, 0.08761756228258717, 3.0840352834902034], [-0.20430175549611906, -4.413928358100152, -1.7617856127879983], [-1.0808312100638977, -1.7908290936555953, -0.026518653338227338], [-0.8851575912846368, -1.2420113973535942, -1.2373533316492336], [-1.82268046636347, -1.115535316258721, 4.364550715361405], [-3.460930068962613, -0.017271217307006204, -0.9222894051907895], [-3.049778110522252, -1.7520692447857318, -0.5868139060856086], [-2.1113264702595242, -4.779443823490021, -4.36415455950815], [0.9404591339551345, -0.9520830071998183, 4.001604420145794], [-4.8581439442919585, -2.3964193250883445, -2.438301104376955], [-2.7107253503261823, -4.132071746786997, -2.6009972239723016], [-3.420036965594741, -2.354359580481696, -3.5025994069929904], [-1.6118745192959372, -0.046330574557696166, 4.573809599421738], [-0.6522316367117407, -2.042284600408266, -2.634194632888987], [-4.295949767597997, -0.43084196822687737, -0.2758996915363463], [-1.1786543705330814, -2.877090012441478, -2.307104930871138], [-0.022039218268002352, -0.4603563310887896, 5.506627280603623], [-2.7190108997654705, -4.4933109274377445, -4.382588593155764], [-3.0659559966952887, -2.284074796898391, -3.784864158529944], [-1.7760959450869898, -3.616744290000429, -1.4902091122165766], [0.04039804635109716, -0.05590841923283404, 5.583065532557033], [-4.79708689579489, -4.647029349076348, -4.057656143595467], [-4.853574807523224, -3.7747848417420844, -1.2922371064895364], [-3.659644121766787, -2.809742636816863, -4.77642723581171], [0.46400582097192705, 0.009715837489138148, 5.464414178585601], [-3.4797061078451335, -3.7314230994743096, -4.633388593345533], [-3.101962854740919, -1.8131207812930574, -4.745834347305024], [-0.13833442578162636, -1.3954354475771504, -2.5639751481603357], [1.3588445344104596, -1.0682829226107526, 4.194758719096439], [-0.6426339667168968, -0.1954640490604449, -2.430895067458727], [-1.5955985610139143, -1.218906323866217, -1.8110807163581755], [-2.2962021405564474, -0.23843096514591355, -2.123962711749081], [0.47457450678810137, -0.8516788997396209, 4.093248876229038], [-3.6794788951228004, -0.3707065071831339, -2.3687932155261224], [-1.6736335269151454, -3.6701099936647816, -3.9235122472527975], [-2.6221032617117097, -1.6148489627142704, -2.6349393637362413], [1.4356444064976972, -0.9347216756164156, 4.277765872588977], [-3.720256105319992, -4.559947881359721, -3.2310579432502546], [-0.37688450651035676, -4.485039316525193, -3.035769919058179], [-3.545614026575273, -2.1766330124162585, -4.104439811153994], [2.9690939890069568, -1.0538678671881005, 5.072324705190996], [-1.4592722724609613, -1.931761424677878, -2.7725565085357546], [-1.2291691731123064, -2.0526158137441017, -0.02750548813222764], [-1.6245193286655284, -4.558415433388024, -4.828134023425222], [0.6925145475164348, -0.005994163789100908, 5.605381161715195], [-2.8090370819760095, -4.748066070071039, -1.5667027276675523], [-0.3661480594476112, -0.7907213467918233, -1.3168725968072446], [-3.666311297045492, -0.8542710139604717, -3.6367807245256607], [2.61401081408322, 1.4126709764062892, -1.0122921444918178], [-1.3944574207331022, -3.0345496128136147, -4.642665094075779], [-2.592878678472772, -2.266533249657879, -4.564376693981615], [-1.7059185823524325, -2.384085510323052, -3.0567661100783647], [0.7167204971627078, -0.013303045386472619, 5.60774278228206], [-0.35455029784882264, -1.316187329751732, -2.5967310839058473], [-0.1591904757516014, -2.400472640613792, -4.009456954944751], [-3.249405021550017, -4.456861991899273, -3.896236090637579], [0.7542174207922986, 0.04134585861291182, 5.620323046819647], [-4.2410858389781305, -1.5634670758622478, -2.781478626947896], [-0.9575754363490177, -2.9498748898720892, -1.396365740226218], [-0.6739624594477354, -2.539550410732787, -1.1269417175356828], [1.6042066463037026, -0.97496292356445, 4.385950530645061], [-2.9423229377428, -0.5886638325731042, -2.4788785688458277], [-3.360939101585246, -4.0143520725645345, -2.1610241703900606], [-2.907865895275492, -3.0298234825487325, -3.8827706002302937], [0.28319130331987946, -1.0256946201406654, 3.5796065584691625], [-2.646074130153436, -0.2669358084559619, -4.348256683445917], [-2.511881152791202, -3.8057190316184437, -0.5574466025101126], [-3.8800560451108774, -3.714357113744075, -2.63976868234374], [-0.19992388266518218, 0.11555182840554633, 5.116969186225387], [-2.1140408635912236, -2.272148698815155, -3.615727664296603], [-4.454415842815355, -1.6913609080024274, -0.5904700894121664], [-1.1990163047080546, -0.45753436424971294, -1.7868028731879457], [3.541322728635186, 0.2511494319424901, 6.650323120486321], [-3.3407992650066367, -0.09137713407622527, -3.6290841753914407], [-1.5083432827543457, -1.6294241186064777, -4.676403622716505], [-1.4825378399401492, -4.877866913518224, -2.8989483841798758], [1.8791459342320158, -1.0978465792558905, 4.469604306841214], [-4.341216757776455, -3.623188421180172, -2.2616583308082427], [-4.248801513388197, -1.7662461899542645, -4.33359063819018], [-4.003747341141915, -0.3209809082396733, -4.637681663603954], [1.3156411741869682, -0.554331533956202, 6.321648703840549], [-4.291471691835013, -0.9844452812685871, -0.5841516872300199], [-4.468183569103418, -1.8195397469524566, -4.34977002544802], [-1.9325391342122873, -4.535504000003067, -3.4704249437407473], [1.3803358455669479, -0.5787503409420706, 6.309545828989181], [-1.7980391336379595, -3.6270824562487087, -4.623904709955184], [-4.558947440767566, -4.085945878890744, -1.631440935726559], [-0.08554419737196817, -2.7625854184068594, -0.561650155428091], [1.3899818414467184, -0.5580425794005122, 6.342343048212614], [-4.027850011009694, -3.7138293015719768, -3.780392861747556], [-4.669518152751549, -0.0020782212271548417, -2.8340282393371203], [-3.621286046780274, -1.3459353685497395, -4.031979563231209], [1.4045493809478453, -0.5168465473882413, 6.329674383597053], [-3.482756577382484, -2.6390815410152086, -3.633284683090129], [-4.260517574984377, -1.6363964654833447, -4.102333993131648], [-1.4171840097909518, -0.6532889630519287, -2.182743422557603], [-1.9526904519832942, 1.0823310305188905, 4.1411252721378995], [-4.608533468536908, -4.697298111999009, -1.2556053610757152], [-0.21858711157169708, -0.8446089018419061, -3.482961933155764], [-0.5168108854550795, -1.9716704989058016, -3.0771512425628593], [2.171170533287498, 1.0279691824012591, 4.908667703575963], [-2.293801028662568, -0.0660073959514147, -0.6546914006114699], [-4.133387240989736, -1.7707205584760324, -4.143494035492907], [-3.6086772303194876, -0.6708379777552134, -1.711145272677581], [-1.8922658800631214, 1.0937490781098207, 4.046106791855489], [-2.8919591669234617, -2.777857309896587, -4.304014975525586], [-1.533480599069331, -1.7186204486795056, -4.148522241126651], [-2.201195247387335, -3.1180252959522297, -1.7574783800842475], [1.2659081836544703, 1.119952094421854, 4.09908347868374], [-4.563450890787981, -3.128019563755947, -2.8770388307039765], [-4.2803827880707495, -4.739974675593642, -4.895968545164113], [-1.3708504083545097, -4.176907581591649, -2.3520866423458227], [-2.433795099724431, 1.102270577838753, 4.325191502622594], [-2.171323451936214, -0.2856525023503398, -1.1874192214374406], [-2.087614081689524, -3.594479527069578, -1.7033884654916385], [-2.2002137522721728, -3.293305584467231, -2.561058357440431], [-1.7711932096267944, 1.1010410595963884, 3.987579479806991], [-1.9935110425063538, -4.019474430366761, -3.1971305468107944], [-0.9154837098335316, -2.638945236822501, -4.667215133386943], [-3.6047146098993315, -1.6063513286918067, -0.6234558592889075], [-1.770595947909954, 1.0963677220244452, 3.9899146881408463], [-4.426824635782948, -4.888629842068439, -1.9426207409168605], [-1.719440475638664, -2.4118752342656253, -0.2859296628637171], [-0.6316671650989125, -2.945766708233642, -0.1468776473149962], [-1.771165027754322, 1.0966941459559312, 4.007465427529099], [-4.820644819535527, -3.0907866297346893, -0.6620824583118123], [-2.1206693236468332, -3.89175981308233, -4.000528446997828], [-2.747603132451749, -3.257000608325896, -2.6370937411035795]]
d2 = [[-0.29928465827728934, -0.6291984196630402], [-0.29928465827728934, -0.6291984196630402], [-0.29928465827728934, -0.6291984196630402], [-0.29928465827728934, -0.6291984196630402], [-2.1822847454863434, -0.38546477857854355], [-2.1822847454863434, -0.38546477857854355], [-2.1822847454863434, -0.38546477857854355], [-2.1822847454863434, -0.38546477857854355], [-0.7314145178510203, -0.40337908456145016], [-0.7314145178510203, -0.40337908456145016], [-0.7314145178510203, -0.40337908456145016], [-0.7314145178510203, -0.40337908456145016], [-0.3308006997028398, -0.5379522560185018], [-0.3308006997028398, -0.5379522560185018], [-0.3308006997028398, -0.5379522560185018], [-0.3308006997028398, -0.5379522560185018], [-0.64207028248235, -0.11274993011633565], [-0.64207028248235, -0.11274993011633565], [-0.64207028248235, -0.11274993011633565], [-0.64207028248235, -0.11274993011633565], [-0.17500695500478025, -0.19260185317337147], [-0.17500695500478025, -0.19260185317337147], [-0.17500695500478025, -0.19260185317337147], [-0.17500695500478025, -0.19260185317337147], [-0.15305533171920488, -0.0843836517626799], [-0.15305533171920488, -0.0843836517626799], [-0.15305533171920488, -0.0843836517626799], [-0.15305533171920488, -0.0843836517626799], [-0.08638591506504553, -0.06028668956500552], [-0.08638591506504553, -0.06028668956500552], [-0.08638591506504553, -0.06028668956500552], [-0.08638591506504553, -0.06028668956500552], [-0.10232716275765137, -0.6023624184001632], [-0.10232716275765137, -0.6023624184001632], [-0.10232716275765137, -0.6023624184001632], [-0.10232716275765137, -0.6023624184001632], [-0.43097222619426306, -0.43803231244166796], [-0.43097222619426306, -0.43803231244166796], [-0.43097222619426306, -0.43803231244166796], [-0.43097222619426306, -0.43803231244166796], [-0.047476585362964945, -0.5193324399715102], [-0.047476585362964945, -0.5193324399715102], [-0.047476585362964945, -0.5193324399715102], [-0.047476585362964945, -0.5193324399715102], [0.9224977605014919, -0.570426122330032], [0.9224977605014919, -0.570426122330032], [0.9224977605014919, -0.570426122330032], [0.9224977605014919, -0.570426122330032], [-0.010803787549087153, -0.05994879768425555], [-0.010803787549087153, -0.05994879768425555], [-0.010803787549087153, -0.05994879768425555], [-0.010803787549087153, -0.05994879768425555], [0.723712860243274, -0.572149016418353], [0.723712860243274, -0.572149016418353], [0.723712860243274, -0.572149016418353], [0.723712860243274, -0.572149016418353], [-0.0046419861614582884, -0.06155030125053673], [-0.0046419861614582884, -0.06155030125053673], [-0.0046419861614582884, -0.06155030125053673], [-0.0046419861614582884, -0.06155030125053673], [0.005400150693652389, -0.045256974129913856], [0.005400150693652389, -0.045256974129913856], [0.005400150693652389, -0.045256974129913856], [0.005400150693652389, -0.045256974129913856], [0.056404393005776154, -0.5348178302688318], [0.056404393005776154, -0.5348178302688318], [0.056404393005776154, -0.5348178302688318], [0.056404393005776154, -0.5348178302688318], [-0.7226837702029022, -0.6004070690182609], [-0.7226837702029022, -0.6004070690182609], [-0.7226837702029022, -0.6004070690182609], [-0.7226837702029022, -0.6004070690182609], [-0.2908244722246904, -0.044453579276371544], [-0.2908244722246904, -0.044453579276371544], [-0.2908244722246904, -0.044453579276371544], [-0.2908244722246904, -0.044453579276371544], [1.0027377546218788, 0.09991343348136172], [1.0027377546218788, 0.09991343348136172], [1.0027377546218788, 0.09991343348136172], [1.0027377546218788, 0.09991343348136172], [0.21139536084906999, -0.6141604232225623], [0.21139536084906999, -0.6141604232225623], [0.21139536084906999, -0.6141604232225623], [0.21139536084906999, -0.6141604232225623], [0.23973093317093513, -0.19322808941736902], [0.23973093317093513, -0.19322808941736902], [0.23973093317093513, -0.19322808941736902], [0.23973093317093513, -0.19322808941736902], [0.25595400239590493, -0.20038599637970997], [0.25595400239590493, -0.20038599637970997], [0.25595400239590493, -0.20038599637970997], [0.25595400239590493, -0.20038599637970997], [0.2609684040626033, -0.19365207938752263], [0.2609684040626033, -0.19365207938752263], [0.2609684040626033, -0.19365207938752263], [0.2609684040626033, -0.19365207938752263], [0.26287142213196657, -0.18283351921315555], [0.26287142213196657, -0.18283351921315555], [0.26287142213196657, -0.18283351921315555], [0.26287142213196657, -0.18283351921315555], [-0.8040336127518917, 0.17252396148384122], [-0.8040336127518917, 0.17252396148384122], [-0.8040336127518917, 0.17252396148384122], [-0.8040336127518917, 0.17252396148384122], [0.34076027665650854, 0.442406560999353], [0.34076027665650854, 0.442406560999353], [0.34076027665650854, 0.442406560999353], [0.34076027665650854, 0.442406560999353], [-0.8283840048806537, 0.18333610205470408], [-0.8283840048806537, 0.18333610205470408], [-0.8283840048806537, 0.18333610205470408], [-0.8283840048806537, 0.18333610205470408], [-0.20966271337976436, 0.49594928370689517], [-0.20966271337976436, 0.49594928370689517], [-0.20966271337976436, 0.49594928370689517], [-0.20966271337976436, 0.49594928370689517], [-0.7964344145393945, 0.14725602148011716], [-0.7964344145393945, 0.14725602148011716], [-0.7964344145393945, 0.14725602148011716], [-0.7964344145393945, 0.14725602148011716], [-0.8354114261662641, 0.195107758611015], [-0.8354114261662641, 0.195107758611015], [-0.8354114261662641, 0.195107758611015], [-0.8354114261662641, 0.195107758611015], [-0.8345826806478053, 0.19375088550263497], [-0.8345826806478053, 0.19375088550263497], [-0.8345826806478053, 0.19375088550263497], [-0.8345826806478053, 0.19375088550263497], [-0.828665737585266, 0.19283315547727867], [-0.828665737585266, 0.19283315547727867], [-0.828665737585266, 0.19283315547727867], [-0.828665737585266, 0.19283315547727867]]


d1 = np.array(d1)
d2 = np.array(d2)
d1 = d1.reshape((33, -1, 3))
d2 = d2.reshape((33, -1, 2))
nb_possibilities = d1.shape[1]
keys = []
for a in range(nb_possibilities):
    for b in range(nb_possibilities):
        for c in range(nb_possibilities):
            for d in range(nb_possibilities):
                for e in range(nb_possibilities):
                    for f in range(nb_possibilities):
                        keys.append(f"{a} {b} {c} {d} {e} {f}")

key2res = {}
for k in keys:
    indices = list(map(int, k.split(" ")))
    d11 = [d1[i, j, :] for i, j in enumerate(indices)]
    d21 = [d2[i, j, :] for i, j in enumerate(indices)]
    d11 = np.array(d11)
    d21 = np.array(d21)
    res = pnp.build.pnp_python_binding.pnp(d11, d21)
    key2res[k] = (res, d11, d21)


for i in range(6, d1.shape[0]):
    d12 = d1[i, :, :]
    d22 = d2[i, :, :]
    deleted = []
    for k in key2res:
        res, d11, d21 = key2res[k]
        select = False
        for ind, (x, y, z) in enumerate(d12):
            xyz = np.array([x, y, z, 1])
            xy = res@xyz
            xy = xy[:3]
            xy /= xy[-1]
            xy = xy[:2]
            diff = np.sum(np.square(xy-d22[ind]))
            if diff < 0.001:
                select = True
                new_xyz = np.array([x, y, z])
                d11 = np.vstack([d11, new_xyz])
                d21 = np.vstack([d21, d22[ind]])
                res = pnp.build.pnp_python_binding.pnp(d11, d21)
                key2res[k] = (res, d11, d21)
                break
        if not select:
            deleted.append(k)

    for k in deleted:
        del key2res[k]
    print(len(key2res))
print(list(key2res.keys()))
for k in key2res:
    res = key2res[k][0]
    print(res)