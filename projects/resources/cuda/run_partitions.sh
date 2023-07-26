# # A100
# DATE=2021_11_02
# N=20000
# DIR=../../../grcuda-data/results/scheduling_multi_gpu/A100/${DATE}_partition_scaling
# mkdir -p ${DIR}
# for g in 1 2 4 8
# do
# 	echo "start ${g} gpu"
# 	for p in 1 2 4 6 8 10 12 16 20 24 28 32
# 	do
# 		echo "${p} partition"
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -r -g 32 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_32_r.csv
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -r -g 64 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_64_r.csv
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -r -g 128 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_128_r.csv
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -g 32 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_32.csv
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -g 64 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_64.csv
# 		bin/b -k b11m -n ${N} -P ${p} -t 10 -p async -g 128 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_128.csv
# 	done
# done

# A100
DATE=2021_11_02
N=8192
DIR=../../../grcuda-data/results/scheduling_multi_gpu/A100/${DATE}_partition_scaling_m13
mkdir -p ${DIR}
for g in 1 2 4
do
	echo "start ${g} gpu"
	for p in 1 2 4 6 8 10 12 16 20 24 28 32
	do
		echo "${p} partition"
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -r -g 6 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_6_r.csv
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -r -g 12 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_12_r.csv
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -r -g 24 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_24_r.csv
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -g 6 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_6.csv
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -g 12 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_12.csv
		bin/b -k b13m -n ${N} -P ${p} -t 5 -p async -g 24 -c 32 -m ${g} | tee ${DIR}/${N}_${g}_${p}_24.csv
	done
done
