#!/usr/bin/env python
__doc__ = \
"""
Customer modeling development
========================================================================
Refering to the isilon directory(e.g., /da/CADD/pqsar/CURRENT/RF_Predictions)
default mtable_dir, e.g., /da/CADD/pqsar/CURRENT/RF_Predictions
index file:
/da/CADD/pqsar/CURRENT/RF_Predictions/MasterTable_indices
RFR predicted master table:
/da/CADD/pqsar/CURRENT/RF_Predictions/MasterTable.csv

input csv file format:

        *** FORMAT one:
                CHIRONID,Qualifier,IC50
                CGA344932,,0.01
                CGP033516A,<,0.01
        OR
        *** if new compounds exist, provide IDs, smiles, Qualifier, and experimental IC50s (uM):
        *** FORMAT two:
                CHIRONID,smiles,Qualifier,IC50
                CGA344932,CO/N=C(/C(C)=NOCc1ccccc1/C(=N\OC)/C(=O)OC)\c2ccc(cc2)OC[Si](C)(C)C,<,0.01
                CGP033516A,CCC1(CCC(=O)NC1=O)c2ccc(cc2)N3C(N)=NC(N)=NC3(C)C,<,0.01
                CGP056089,COc1ccc4c(c1)c2cc(OC)c(cc2c5CC3CCCN3Cc45)OC,<,0.01
"""

def get_job_id(process):
        job_sub = process.stdout.decode('utf-8')
        try:
                jobID = re.search('Your job-array (\d.+?)[ \.]', job_sub)[1]
        except:
                jobID = re.search('Your job (\d.+?)[ \.]', job_sub)[1]

        return(jobID)


def bash_head(job_name, job_num=1, job_mem='4G', job_time=32000, threads=1, py=False):
        # generate the first few lines of bash file
        bash_line = '#$ -S /bin/bash\n'
        bash_line += '#$ -N {}_{}\n'.format(job_name, job_num)
        bash_line += '#$ -cwd\n'
        bash_line += '#$ -o o_{}.txt\n'.format(job_name)
        bash_line += '#$ -e e_{}.txt\n'.format(job_name)
        bash_line += '#$ -q default.q\n'
        bash_line += '#$ -l h_rt={}\n'.format(job_time)
        bash_line += '#$ -j y\n'
        bash_line += '#$ -V\n'
        bash_line += '#$ -t 1:{}\n'.format(job_num)
        #bash_line += '#$ -tc 200\n'
        bash_line += '#$ -l m_mem_free={}\n'.format(job_mem)
        bash_line += '#$ -pe smp {}\n'.format(threads)
        #bash_line += '#$ -binding linear:{}\n\n'.format(threads)
        bash_line += 'export OMP_NUM_THREADS={}\n'.format(threads)
        bash_line += 'export OPENBLAS_NUM_THREADS={}\n'.format(threads)
        bash_line += 'export MKL_NUM_THREADS={}\n\n'.format(threads)

        if py:
                bash_line += 'module unload python\n'
                bash_line += 'module unload anaconda\n'
                bash_line += "DISTR=`lsb_release -a | grep Distributor | awk -F' ' '{print $3}'`\n"
                bash_line += 'REDHAT=RedHatEnterpriseServer\n'
                bash_line += 'if [ "$DISTR" = "$REDHAT" ]\n'
                bash_line += 'then\n'
                bash_line += 'module use /usr/prog/anaconda/modules\n'
                bash_line += 'else\n'
                bash_line += 'module use /usr/prog/sigma/modules\n'
                bash_line += 'fi\n'
                bash_line += 'module load PythonDS/v0.8\n'
        return(bash_line)

def paste_pred(tmp_dir, current_dir, mfile):
        # paste all the predict file into one
        # first use 90 cluster nodes and then just one node
        init = list(set([f[0 : 2] for f in os.listdir(tmp_dir) if f.endswith('_RFpred_tmp.txt')]))
        jobIDs = []

        for i in init:
                bash_line = bash_head(job_name='PST1', job_mem='8G')
                #bash_line += "awk 'FNR==1{f++}{a[f,FNR]=$3}END{for(x=1;x<=FNR;x++){for(y=1;y<ARGC;y++)printf(\"%s \",a[y,x]);print ""}}'" + " {}*_RFpred_10001.txt > {}/F_{}.txt".format(i, pred_dir, i)
                bash_line += "awk " + "'" + "{ a[FNR] = (a[FNR] ? a[FNR] FS : \"\") $2 } END { for(i=1;i<=FNR;i++) print a[i] }" + "'" + " {}/{}*pred_tmp.txt > {}/merge_round_one_{}_tmp.txt\n".format(tmp_dir, i, tmp_dir, i)
                bash_file = '{}/merge_round_one_{}_tmp.sh'.format(tmp_dir, i)
                with open(bash_file, 'w') as fid:
                        fid.writelines(bash_line)


                process = subprocess.run(["qsub", "-l", "m_mem_free=" + '4G', bash_file], stdout=subprocess.PIPE)
                jobID = get_job_id(process)
                jobIDs.append(jobID)

        current_jobs(jobIDs)

        bash_file = 'merge_round_two_tmp.sh'
        bash_line = bash_head(job_name='PST2', job_mem='16G')
        bash_line += "paste -d ' ' {}/merge_round_one_*_tmp.txt > {}/{}\n".format(tmp_dir, current_dir, mfile)

        with open('{}/{}'.format(tmp_dir, bash_file), 'w') as fid:
                fid.writelines(bash_line)

        process = subprocess.run(["qsub", "-l", "m_mem_free=16G",       tmp_dir + "/" +bash_file], stdout=subprocess.PIPE)
        jobID = get_job_id(process)
        current_jobs([jobID])


def current_jobs(jobIDs):
        while True:
                jobs = subprocess.Popen("qstat | tail -n +3 | awk -F' ' '{print $1}' | sort -u", shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
                jobs.remove('')
                running = subprocess.Popen("qstat | tail -n +3 | awk -F' ' '{print $5}' | sort -u", shell=True, stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
                running.remove('')

                if len(set(jobIDs).intersection(set(jobs))) == 0:
                        break

                if len(running) == 1 and running[0] == 'dr':
                        break
                else:
                        time.sleep(60)


class QueryCmpd:

        def __init__(self, mtable_dir):
                # initialize indices and MasterTable files
                self.index_file = '{}/{}'.format(mtable_dir, [fname for fname in os.listdir(mtable_dir) if fname.endswith('_indices')][0])
                self.MasterTable = '{}/MasterTable.csv'.format(mtable_dir)

                self.indices = joblib.load(self.index_file)
                self.fid = open(self.MasterTable)
                self.separator = ',' if self.MasterTable.endswith('.csv') else '\t'

        def columns(self):
                self.fid.seek(0, 0)
                line = self.fid.readline().strip().split(self.separator)
                col = line[1: ]

                return(col)

        def get(self, cmpd, raw=False):
                index = self.indices.loc[cmpd].values[0]
                self.fid.seek(index, 0)


                line = self.fid.readline().strip().split(self.separator)
                line_name = line[0]

                if raw:
                        return(line_name, line[1: ])

                line_data = [float(x) for x in line[1: ]]

                return(line_name, line_data)


def rf_tmp_mdl_dev(rf_mdl_py):
        # create temporary python scripts for RF prediction
        py_line = 'import sys\n'
        py_line += 'import pandas as pd\n'
        py_line += 'import numpy as np\n'
        py_line += 'from sklearn.ensemble import RandomForestRegressor\n'
        py_line += 'import joblib\n\n'

        py_line += 'data = joblib.load(sys.argv[2])\n'
        py_line += 'model = joblib.load(sys.argv[3])\n'
        py_line += "data[sys.argv[1]] = model.predict(pd.DataFrame([np.array(x) for x in data['FP']]))\n"

        py_line += "data = data[sys.argv[1]]\n"
        py_line += "data.round(3).to_csv(sys.argv[4], sep='\t', index=True, header=True)\n"

        with open(rf_mdl_py, 'w') as fid:
                fid.writelines(py_line)


def transform_PIC50(df_cmpd):
        #
        #
        df_duplicated = df_cmpd.loc[df_cmpd.duplicated(subset=['CHIRONID'], keep=False)]
        df_unique = df_cmpd.drop_duplicates(subset=['CHIRONID'], keep=False)

        if len(df_duplicated) > 0:
                cpds = set(df_duplicated['CHIRONID'])

                df_dedu = pd.DataFrame(0, index=cpds, columns=df_duplicated.columns)

                for cpd in cpds:
                        df_tmp = df_duplicated.loc[df_duplicated['CHIRONID'] == cpd]
                        qualifier = set(df_tmp['Qualifier'])

                        qlf = np.nan

                        if len(qualifier) == 1:
                                if np.nan in qualifier or '=' in qualifier:
                                        PIC50 = np.exp(np.mean(np.log(df_tmp['PIC50'])))
                                elif '>' in qualifier:
                                        PIC50 = np.max(df_tmp['PIC50'])
                                        qlf = '>'
                                elif '<' in qualifier:
                                        PIC50 = np.min(df_tmp['PIC50'])
                                        qlf = '<'
                        elif len(qualifier) == 2:
                                if np.nan in qualifier:
                                        PIC50 = np.mean(df_tmp.loc[df_tmp['Qualifier'] == np.nan, 'PIC50'])
                                else:
                                        PIC50 = np.nan
                        else:
                                PIC50 = np.nan

                        #df_tmp.loc[df_tmp.index, 'PIC50'] = PIC50

                        df_dedu.loc[cpd] = df_tmp.iloc[0, :]
                        df_dedu.loc[cpd, 'PIC50'] = PIC50
                        df_dedu.loc[cpd, 'Qualifier'] = qlf

                df_dedu = df_dedu.loc[df_dedu['PIC50'].notnull()]

                df_unique = pd.concat([df_unique, df_dedu], ignore_index=True)

        df_unique = df_unique.loc[df_unique['PIC50'] > 0]

        df_unique['PIC50'] = -np.log10(df_unique['PIC50'] * 1e-6)
        Qualifiers = set(df_unique['Qualifier'])

        if '>' in Qualifiers:
                df_unique.loc[df_unique['Qualifier'] == '>', 'PIC50'] = df_unique.loc[df_unique['Qualifier'] == '>', 'PIC50'] - 1
        if '<' in Qualifiers:
                df_unique.loc[df_unique['Qualifier'] == '<', 'PIC50'] = df_unique.loc[df_unique['Qualifier'] == '<', 'PIC50'] + 1

        df_unique.drop(['Qualifier'], axis=1, inplace=True)

        if 'smiles' in df_unique.columns:
                df_unique_pIC50 = df_unique.groupby('smiles', as_index=False).mean()
                df_unique = df_unique.groupby('smiles', as_index=False).nth(0).copy()
                df_unique.drop('PIC50', axis=1, inplace=True)
                df_unique = df_unique.merge(df_unique_pIC50, how='right', right_on='smiles', left_on='smiles')
                df_unique = df_unique[['CHIRONID', 'PIC50', 'smiles']]
        df_unique.set_index('CHIRONID', inplace=True)

        return df_unique



def survey(input_file, sys_argv):
        survey_csv = '/da/CADD/pqsar/Survey/survey.csv'
        #host_name = os.getenv('HOSTNAME')
        host_name = os.uname()[1]
        count = subprocess.run(["wc", "-l", input_file], stdout=subprocess.PIPE).stdout.decode('utf-8').split()[0]
        count = int(count) - 1

        localtime = time.localtime()
        localtime = "{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(localtime[0], localtime[1], localtime[2], localtime[3], localtime[4], localtime[5])

        with open(survey_csv, 'a+') as fid:
                fid.writelines(localtime + ',')
                fid.writelines(host_name + ',')
                fid.writelines(' '.join(sys_argv) + ',')
                fid.writelines('{},{}\n'.format(input_file, count))


###################################################################################################
def main():

        parser = argparse.ArgumentParser(
                                                description=description,
                                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                                epilog=epilog)


        parser.add_argument('-i', '--Input', help='Input csv file', metavar='*.csv / *.txt')
        parser.add_argument('-chembl', '--ChEMBL', help='Build custom models using ChEMBL profile', type=str, default='False', metavar='True/False')
        parser.add_argument('-s', '--Skip', help='Skip converting IC50 (uM) to pIC50: True/ False', type=bool, default=False, metavar='True/False')
        parser.add_argument('-a', '--Assay', help='Assay ID to be excluded from modeling', metavar='assay ID')
        parser.add_argument('-o', '--Output', help='Output folder')


        if len(sys.argv) < 5:
                parser.print_help()
                sys.exit(1)

        args = parser.parse_args()

        if not args.Input:
                raise IOError('No input compound')
        argv1 = args.Input
        #argv1 = '/home/zhuxi1q/customer_model/customer_cmpd.csv'
        survey(args.Input, sys.argv)

        if not args.Assay:
                raise IOError('No input assay CDS ID')
        assay = args.Assay
        #assay = '10001'

        current_dir = args.Output if args.Output else os.getcwd()

        if not os.path.isdir(current_dir):
                current_dir = os.getcwd()
                print('Warning: {} doesn\'t exist, output directory is set as {} '.format(args.Output, current_dir))
        else:
                current_dir = current_dir.rstrip('/')

        # hard code for now
        smiles = 'smiles'
        molecule = 'molecule'
        # use this line for production
        if args.ChEMBL.lower() in ['true', 't', 'yes', 'y']:
                argv2 = '/da/CADD/pqsar/chembl_28'
        else:
                argv2 = '/da/CADD/pqsar/CURRENT'
        # just for test
        #argv2 = '/home/zhuxi1q/CURRENT08'

        nvs_path = '/da/CADD/pqsar/Data'
        mtable_dir = '{}/RF_Predictions'.format(argv2)
        mfile = '{}_msplit_tmp.txt'.format(assay)
        py_dir = '/da/CADD/pqsar/SCRIPTS/Backup-Apr2021'
        # temporary folder to store the intermediate files
        tmp_dir = '{}/tmp_folder_4customer_model_delete_after'.format(current_dir)

        ###################################################################################################
        if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
                os.mkdir(tmp_dir)
        else:
                os.mkdir(tmp_dir)

        seperator = ',' if argv1.endswith('.csv') else '\t'

        df_cmpd = pd.read_csv(argv1, index_col=None, dtype={'Id':'string'},  header=0, sep=seperator)
        col_names = list(df_cmpd.columns)


        if len(col_names) == 3:
                col_names = ['CHIRONID', 'Qualifier', 'PIC50']
                df_cmpd.columns = col_names
        elif len(col_names) == 4:
                col_names = ['CHIRONID', 'smiles','Qualifier', 'PIC50']
                df_cmpd.columns = col_names
                df_cmpd = df_cmpd[['CHIRONID', 'Qualifier', 'PIC50', 'smiles']]
        else:
                print('Error. Need 3 or 4 columns')
                sys.exit(1)

        if args.Skip == False:
                print('\tBefore quality check and activity converting: {} compounds'.format(len(df_cmpd)))
                df_cmpd = transform_PIC50(df_cmpd)
                df_cmpd.sort_index(axis=0, ascending=True, inplace=True)
                print('\tAfter quality check and activity converting: {} compounds'.format(len(df_cmpd)))
        else:
                df_cmpd.drop(['Qualifier'], axis=1, inplace=True)
                df_cmpd.set_index('CHIRONID', inplace=True)
                print('\tDataset has {} compounds'.format(len(df_cmpd)))

        df_cmpd.dropna(axis=1, how='any')
        col_names = list(df_cmpd.columns)

        if len(col_names) == 1:
                FLAG = False
        elif len(col_names) >= 2:
                FLAG = True

        cmpd_ids = list(df_cmpd.index)

        # get the index file. MasterTable_indices
        # the suffix _indices should be kept the same. Should be only one _indices file in one folder
        if not os.path.isfile ('{}/{}'.format(current_dir, mfile)):
                if FLAG == False:
                        print('\tStart retrieving compound\'s profile')
                        p = QueryCmpd(mtable_dir)
                        p_col = p.columns()

                        cmpd_profile = pd.DataFrame(0.0, index=cmpd_ids, columns=p_col)

                        # list of new compounds
                        cmpd_new = []

                        for cmpd in cmpd_ids:
                                # retrieve compound's profile
                                try:
                                        name, pqsar_vec = p.get(cmpd)
                                        assert name == cmpd
                                        cmpd_profile.loc[cmpd] = [float(s) for s in pqsar_vec]
                                except:
                                        #cmpd_profile.drop(cmpd, axis=0, inplace=True)
                                        cmpd_new.append(cmpd)

                        cmpd_profile.drop(cmpd_new, axis=0, inplace=True)
                        new_len = len(cmpd_new)
                        nvs_len = len(cmpd_profile)
                        print('\tOnly {} compounds will be predicted'.format(nvs_len))
                        #if nvs_len == 0:
                        #       raise IOError('\tNo single compound was found')

                        if new_len != 0:
                                print('\tWarning! {} compounds not found : {}'.format(new_len, cmpd_new))


                        else:
                                print('\tAll compounds ({}) prediction. \n\tMight take 20 min to get the results'.format(len(cmpd_ids)))
                                FLAG = False


                        if len(col_names) == 1:
                                ## add smiles
                                if args.ChEMBL.lower() in ['true', 't', 'yes', 'y']:
                                        nvs_cmpds = '{}/chembl_28_AllAssayCmpd.csv'.format(nvs_path)
                                else:
                                        months = ['12', '11', '10', '09', '08', '07', '06', '05', '04', '03', '02', '01']

                                        for m in months:
                                        # refer to the file of All NVS compounds and find the latest file.
                                                nvs_cmpds = '{}/AllMagmaComp_2021_{}.csv'.format(nvs_path, m)
                                                if os.path.isfile(nvs_cmpds):
                                                        break

                                nvs_cmpds_info = pd.read_csv(nvs_cmpds, dtype={'CDS_ID':'string'}, header=0, index_col=0)
                                nvs_cmpds_info.columns = ['smiles']

                                idx = df_cmpd.index
                                df_cmpd = df_cmpd.merge(nvs_cmpds_info, how='left', left_index=True, right_index=True)

                if not FLAG:
                        # save the msplit file
                        cmpd_profile = cmpd_profile.merge(df_cmpd, how='inner', left_index=True, right_index=True)
                        cmpd_profile.index.name = 'PREFERRED_NAME'
                        #cmpd_profile.round(3).to_csv('{}/{}'.format(current_dir, mfile), sep='\t')
                        data_len = len(cmpd_profile)
                        cmpd_profile.to_csv('{}/{}'.format(current_dir, mfile), sep='\t')

                        del cmpd_profile
                        del df_cmpd
                else:
                        print('\tAll compounds ({}) RF prediction. \n\tMight take more than 2 hours to get the results'.format(len(cmpd_ids)))
                        # go to the directory of RF models and get the CDS IDs of all models.
                        rf_dir = '{}/RF_Models'.format(argv2)
                        rf_models = [f for f in os.listdir(rf_dir) if f.endswith('.RFmodel')]
                        rf_cds_id = [m.split('.')[0] for m in rf_models]

                        print('\tStart fingerprint computing')
                        try:
                                PandasTools.AddMoleculeColumnToFrame(df_cmpd, smiles, molecule)
                        except:
                                print("Erroneous exception was raised and captured...")

                        #remove records with empty pIC50s
                        df_cmpd = df_cmpd.loc[df_cmpd['PIC50'].notnull()]
                        #remove records with empty molecules
                        df_cmpd = df_cmpd.loc[df_cmpd[molecule].notnull()]
                        #df_cmpd['FP'] = [pQSAR.computeFP(m) for m in df_cmpd[molecule]]
                        df_cmpd['FP'] = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in df_cmpd[molecule]]
                        tmp_fps = '{}/tmp_fp.joblib'.format(tmp_dir)
                        joblib.dump(df_cmpd, tmp_fps)

                        rf_mdl_py = '{}/rf_prediction_tmp.py'.format(tmp_dir)
                        rf_tmp_mdl_dev(rf_mdl_py)

                        # need this file to get the number of molecules of each assay
                        df_summary = pd.read_csv('{}/Summary_pQSAR/SummaryRF.csv'.format(argv2), index_col=0, dtype={'CDS_ID':'string'}, header=0)
                        #print(df_summary.columns)
                        #print(df_summary.dtypes)
                        df_summary.index = [str(d) for d in df_summary.index]
                        df_info_all = df_summary[['Count(molecules)']]
                        df_info_all.reset_index(inplace=True)
                        df_info_all=df_info_all.rename(columns = {'index':'CDS_ID'})
                        df_info_all['CDS_ID'].map(str)
                        df_info_all.set_index('CDS_ID', inplace=True)
                        #print(df_info_all.head)
                        #print(df_info_all.dtypes)
                        while True:
                                # in case some prediction failed. run it until get all the prediction
                                df_info = df_info_all.loc[rf_cds_id]
                                print(df_info.dtypes)
                                # class assays in four categories and assign them different memories.
                                df_info_50k = df_info.loc[df_info['Count(molecules)'] >= 50_000]
                                df_info_10k = df_info.loc[(df_info['Count(molecules)'] >= 10_000) & (df_info['Count(molecules)'] <= 50_000)]
                                df_info_5k = df_info.loc[(df_info['Count(molecules)'] >= 5000) & (df_info['Count(molecules)'] < 10_000)]
                                df_info_2k = df_info.loc[(df_info['Count(molecules)'] >= 2000) & (df_info['Count(molecules)'] < 5000)]
                                df_info_s2k = df_info.loc[df_info['Count(molecules)'] < 2000]

                                df_info_50k.to_csv('{}/assay_50k.txt'.format(tmp_dir), sep='\t', index=True, header=False)
                                df_info_10k.to_csv('{}/assay_10k.txt'.format(tmp_dir), sep='\t', index=True, header=False)
                                df_info_5k.to_csv('{}/assay_5k.txt'.format(tmp_dir), sep='\t', index=True, header=False)
                                df_info_2k.to_csv('{}/assay_2k.txt'.format(tmp_dir), sep='\t', index=True, header=False)
                                df_info_s2k.to_csv('{}/assay_s2k.txt'.format(tmp_dir), sep='\t', index=True, header=False)

                                assay_class = ['assay_50k.txt', 'assay_10k.txt', 'assay_5k.txt', 'assay_2k.txt', 'assay_s2k.txt']
                                assay_num_in_class = [len(df_info_50k), len(df_info_10k), len(df_info_5k), len(df_info_2k), len(df_info_s2k)]
                                m_mem = ['8G', '6G', '4G', '2G', '1G']
                                print('\tStart RFR ({} models) prediction'.format(len(rf_cds_id)))

                                assay_class = [a for i, a in enumerate(assay_class) if assay_num_in_class[i] > 0]
                                m_mem = [m for i, m in enumerate(m_mem) if assay_num_in_class[i] > 0]
                                assay_num_in_class = [a for a in assay_num_in_class if a > 0]

                                jobIDs = []

                                for i, ac in enumerate(assay_class):
                                        # submit job for each category
                                        # use ModelPredictions.py to predict molecule's activity using RF models
                                        bash_line = bash_head(job_name='RF', job_num=assay_num_in_class[i], job_mem=m_mem[i], py=True)
                                        bash_line += "SAMPLES_LIST='{}/{}'\n".format(tmp_dir, ac)
                                        bash_line += "ARGS=`sed -n \"${SGE_TASK_ID}p\" < $SAMPLES_LIST | awk -F'\\t' '{print $1}'`\n"

                                        bash_line += ' '.join(['python', rf_mdl_py, '${ARGS[0]}', tmp_fps, '{}/RF_Models/{}.RFmodel'.format(argv2, '${ARGS[0]}'), '{}/{}_RFpred_tmp.txt'.format(tmp_dir, '${ARGS[0]}')])

                                        bash_file = '{}/rf_pred_{}_tmp.sh'.format(tmp_dir, ac)
                                        with open(bash_file, 'w') as fid:
                                                fid.writelines(bash_line)

                                        process = subprocess.run(["qsub", "-l", "h_rt=32000" + ",m_mem_free=8G", bash_file, "/dev/null"], stdout=subprocess.PIPE)
                                        jobID = get_job_id(process)
                                        jobIDs.append(jobID)

                                current_jobs(jobIDs)
                                success = [t.split('_')[:-2] for t in os.listdir(tmp_dir) if t.endswith('_RFpred_tmp.txt')]
                                success = ['_'.join(success[i]) for i in range(len(success))]
                                #print(set(success))
                                #print('*******************************************************************************{rf_cds_id}********************************************************************************************')
                                #print(set(rf_cds_id))
                                failed = set(rf_cds_id) - set(success)
                                print((len(rf_cds_id), len(success)))
                                print(len(failed))
                                if len(failed) > 0:
                                        print('\t{} RFR models failed, will try again'.format(len(failed)))
                                        rf_cds_id = list(failed)

                                else:
                                        break

                        print('\tRFR prediction finished')

                        # collect all the prediction into msplit file
                        print('\tStart merging prediction')
                        paste_pred(tmp_dir, current_dir, mfile)
                        cmpd_profile = pd.read_csv('{}/{}'.format(current_dir, mfile), sep=' ', index_col=None, dtype={'CDS_ID':'string'}, header=0)
                        cmpd_profile.index = df_cmpd.index
                        df_cmpd.drop(['FP', molecule], axis=1, inplace=True)
                        cmpd_profile = cmpd_profile.merge(df_cmpd, how='inner', left_index=True, right_index=True)
                        cmpd_profile.index.name = 'PREFERRED_NAME'
                        data_len = len(cmpd_profile)
                        cmpd_profile.round(3).to_csv('{}/{}'.format(current_dir, mfile), sep='\t')

                        del cmpd_profile
                        del df_cmpd

        else:
                wc_line = subprocess.run(["wc", "-l", "{}/{}".format(current_dir, mfile)], stdout=subprocess.PIPE).stdout.decode('utf-8').split()[0]
                data_len = int(wc_line) - 1


        # PLS model development
        print('\tStart building PLS model for {} compounds'.format(data_len))
        jobIDs = []
        if data_len >= 80_000:
                bash_line = bash_head(job_name='PLS', job_num=1, job_mem='96G', py=True)
        elif 10_000 < data_len < 80_000:
                bash_line = bash_head(job_name='PLS', job_num=1, job_mem='36G', py=True)
        else:
                bash_line = bash_head(job_name='PLS', job_num=1, job_mem='16G', py=True)

        bash_line += ' '.join(['python', '{}/ModelBuildingPLS.py'.format(py_dir), 'False', '{}/{}'.format(current_dir, mfile), 'PIC50', assay, '{}/{}.PLSmodel'.format(current_dir, assay), '{}/{}_info.txt'.format(current_dir, assay), '{}/{}_predicted.txt'.format(current_dir, assay), '0.75', 'True\n'])
        #bash_line += 'python ' + PythonScriptFN + ' ' +  MasterTableDir + ' ' + resultsDir + '/' + '${ARGS}' + split + ' ' + AssayColumnName + ' ${ARGS} ' + resultsDir + '/' + '${ARGS}' + '.' + PLS_model_suffix + ' ' + resultsDir + '/' + '${ARGS}' + info + ' ' + resultsDir + '/' + '${ARGS}' + predictedVexp + ' ' + str(fractio2test) + '\n'
        bash_file = '{}/pls_model_tmp.sh'.format(tmp_dir)

        with open (bash_file, 'w') as fid:
                fid.writelines(bash_line)

        process = subprocess.run(["qsub", "-l", "h_rt=160000", bash_file, "/dev/null"], stdout=subprocess.PIPE)

        jobID = get_job_id(process)
        current_jobs([jobID])
        subprocess.run(["rm", "-r", tmp_dir], stdout=subprocess.PIPE)
        print('\tCustom model might be successfully built here {}/'.format(current_dir))

if __name__ == "__main__":

        import os, sys
        if sys.version_info[0 : 3] != (3, 8, 6):
                print("************************************************************************")
                print("****  Warning: Please load PythonDS/v0.8 by running script below  ******")
                print("************************************************************************")
                print("export MODULEPATH=${MODULEPATH}:/usr/prog/modules/all\nmodule use /usr/prog/modules/all\nmodule load uge\nmodule load PythonDS/v0.8\n")
                sys.exit(0)
        import time
        import shutil
        import re
        import argparse
        import subprocess
        import pandas as pd
        import numpy as np
        #import CommonTools as pQSAR
        from rdkit.Chem import AllChem
        from rdkit.Chem import PandasTools
        import joblib
        main()
