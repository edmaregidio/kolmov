from kolmov import  crossval_table, fit_table
import collections
import argparse
import os
parser = argparse.ArgumentParser(description = '', add_help = False)

parser.add_argument('--tunedFiles',action='store', dest='tunedFiles', required=True, help = "path to tuned files (.pic.gz)")
parser.add_argument('--modelTag', action='store', dest='modelTag', required=True, help = "model tag (e.g v1.mc16)")
parser.add_argument('--dataFiles',action='store',dest='dataFiles',required=True, help = "path to npz files")
parser.add_argument('--refFiles', action='store', dest='refFiles', required = True, help = "path to reference files (.ref.pic.gz")
parser.add_argument('--outputPath', action='store', dest='outputPath', required = True, help = "path to the output files")
parser.add_argument('--signature', action='store', dest='signature', required = True, help = "particle signature (Electron, Photon, Muon)")

origin_path = os.getcwd()
etbins = [15,20,30,40,50,100000]
etabins = [0, 0.8 , 1.37, 1.54, 2.37, 2.5]
tuned_info = collections.OrderedDict( {
                  # validation
                  "max_sp_val"      : 'summary/max_sp_val',
                  "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
                  "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
                  # Operation
                  "max_sp_op"       : 'summary/max_sp_op',
                  "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
                  "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
                  } )
args = parser.parse_args()

cv  = crossval_table( tuned_info, etbins = etbins , etabins = etabins )
cv.fill(args.tunedFiles+'/*/*.gz',args.modelTag)
best_inits = cv.filter_inits("max_sp_val")
best_inits = best_inits.loc[(best_inits.model_idx==0)]
best_sorts = cv.filter_sorts(best_inits, 'max_sp_val')
best_models = cv.get_best_models(best_sorts, remove_last=True)


def generator( path ):
    def norm1( data ):
        norms = np.abs( data.sum(axis=1) )
        norms[norms==0] = 1
        return data/norms[:,None]
    from Gaugi import load
    import numpy as np
    d = load(path)
    feature_names = d['features'].tolist()
    
    n = d['data'].shape[0]

     # extract all shower shapes
    data_reta   = d['data'][:, feature_names.index('L2Calo_reta')].reshape((n,1)) / 1.0
    data_eratio = d['data'][:, feature_names.index('L2Calo_eratio')].reshape((n,1)) / 1.0
    data_f1     = d['data'][:, feature_names.index('L2Calo_f1')].reshape((n,1)) / 0.6
    data_f3     = d['data'][:, feature_names.index('f3')].reshape((n,1)) / 0.04
    data_weta2  = d['data'][:, feature_names.index('weta2')].reshape((n,1)) / 0.02
    data_wstot  = d['data'][:, feature_names.index('wtots1')].reshape((n,1)) / 1.0

    #target = d['target']

  # Fix all shower shapes variables
    print( 'eratio > [10,inf[ = %d'%len(data_eratio[data_eratio>10.0]) )
    data_eratio[data_eratio>10.0]=0
    print( 'eratio > [1,10[ = %d'%len(data_eratio[data_eratio>1.0]) )
    data_eratio[data_eratio>1.]=1.0
    print ('wstor < -99 =  %d'%len(data_wstot[data_wstot<-99]))
    data_wstot[data_wstot<-99]=0

     # This is mandatory
    #splits = [(train_index, val_index) for train_index, val_index in cv.split(data_reta,target)]
    dataRings = norm1(d['data'][:,1:101])
    data_shower = np.concatenate( (data_reta,data_eratio,data_f1,data_f3,data_weta2, data_wstot), axis=1)
     #dataSS = np.transpose(data_shower)
    data = np.concatenate((dataRings,data_shower),axis=1)
    
    #data = norm1(d['data'][:,1:101])
    target = d['target']
    avgmu = d['data'][:,0]
    references = ['T0HLTElectronT2CaloTight','T0HLTElectronT2CaloMedium','T0HLTElectronT2CaloLoose','T0HLTElectronT2CaloVLoose']
    ref_dict = {}
    for ref in references:
        answers = d['data'][:, feature_names.index(ref)]
        signal_passed = sum(answers[target==1])
        signal_total = len(answers[target==1])
        background_passed = sum(answers[target==0])
        background_total = len(answers[target==0])
        pd = signal_passed/signal_total
        fa = background_passed/background_total
        ref_dict[ref] = {'signal_passed': signal_passed, 'signal_total': signal_total, 'pd' : pd,
                             'background_passed': background_passed, 'background_total': background_total, 'fa': fa}

    return data, target, avgmu

fileName = os.listdir(args.dataFiles)[0]
model_tag = fileName[0:[n for n in range(len(fileName)) if fileName.find('_et', n) == n][-1]-1]
refName = os.listdir(args.refFiles)[0]
path = args.dataFiles + model_tag + '{ET}_eta{ETA}.npz'
ref_tag = refName[0:[n for n in range(len(fileName)) if fileName.find('_et', n) == n][-1]-1]
ref_path = args.refFiles + ref_tag + '{ET}_eta{ETA}.ref.pic.gz'

paths = [[ path.format(ET=et,ETA=eta) for eta in range(5)] for et in range(5)]
ref_paths = [[ ref_path.format(ET=et,ETA=eta) for eta in range(5)] for et in range(5)]
ref_matrix = [[ {} for eta in range(5)] for et in range(5)]
references = ['T0HLTElectronT2CaloTight','T0HLTElectronT2CaloMedium','T0HLTElectronT2CaloLoose','T0HLTElectronT2CaloVLoose']
references = ['tight_cutbased', 'medium_cutbased' , 'loose_cutbased', 'vloose_cutbased']

from saphyra.core import ReferenceReader
for et_bin in range(5):
    for eta_bin in range(5):
        for name in references:
            refObj = ReferenceReader().load(ref_paths[et_bin][eta_bin])
            pd = refObj.getSgnPassed(name)/refObj.getSgnTotal(name)
            fa = refObj.getBkgPassed(name)/refObj.getBkgTotal(name)
            ref_matrix[et_bin][eta_bin][name] = {'pd':pd, 'fa':fa}




# get best models
ct  = fit_table( generator, etbins , etabins, 0.02, 1.5, 16, 60 )
os.chdir(args.outputPath)
os.mkdir(args.outputPath + 'exportToolOutput/')
os.mkdir(args.outputPath + 'exportToolOutput/models')
ct.fill(paths, best_models, ref_matrix,'exportToolOutput/plots')
os.chdir(origin_path)

table = ct.table()
ct.dump_beamer_table(table, best_models, 'test', 'test')
ct.export(best_models, model_tag[0:len(model_tag)-3]+'.model_v1.electronLoose.et%d_eta%d', args.signature+'RingerLooseTriggerConfig.conf', 'loose_cutbased', to_onnx=True)
ct.export(best_models, model_tag[0:len(model_tag)-3]+'.model_v1.electronMedium.et%d_eta%d', args.signature+'RingerMediumTriggerConfig.conf', 'medium_cutbased', to_onnx=True)
ct.export(best_models, model_tag[0:len(model_tag)-3]+'.model_v1.electronTight.et%d_eta%d', args.signature+'RingerTightTriggerConfig.conf', 'tight_cutbased', to_onnx=True)
ct.export(best_models, model_tag[0:len(model_tag)-3]+'.model_v1.electronVeryLoose.et%d_eta%d', args.signature+'RingerVeryLooseTriggerConfig.conf', 'vloose_cutbased', to_onnx=True)

commandModels = 'mv *.h5 *.json *.onnx ' + args.outputPath + 'exportToolOutput/models'
os.system(commandModels)
commandConf = 'mv *.conf ' + args.outputPath + 'exportToolOutput/'
os.system(commandConf)
