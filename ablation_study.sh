if [ $# -ge 3 ]
then
    CUDA_VISIBLE_DEVICES=$1
    dataset_name=$2
    model_type=$3
else
    echo "Incorrect number of parameters."
    exit
fi

partial_command="-${model_type}"

if [ ${model_type} == "sci" ]
then
    partial_model_name="sci_sft-rb"

elif [ ${model_type} == "scs" ]
then
    partial_model_name="ift-rb_scs"

elif [ ${model_type} == "nml" ]
then
    partial_model_name="ift-rb_sft-rb_nml"

elif [ ${model_type} == "sum" ]
then
    partial_model_name="ift-add_sft-add"
    partial_command="-ift add -sft add"

elif [ ${model_type} == "rate_sum" ]
then
    partial_model_name="ift-rate_sft-rate"
    partial_command="-ift rate -sft rate"

elif [ ${model_type} == "none" ]
then
    partial_model_name="sci_scs"
    partial_command="-sci -scs"

elif [ ${model_type} == "full" ]
then
    partial_model_name="ift-rb_sft-rb_full"
    partial_command="-ne 100"
#    partial_model_name="ift-bl_sft-bl"
#    partial_command="-ift bilinear -sft bilinear"

else
    echo "Incorrect model type."
fi

if [ ${dataset_name} == "cais" ]
then
    bies=$4
    tokenizer=$5

    if [ ${tokenizer} == "ltp" ]
    then
        dataDir=data/cais_bies-${bies}_token-True
    elif [ ${tokenizer} == "jieba" ]
    then
        dataDir=data/cais_bies-${bies}_token-jieba
    elif [ ${tokenizer} == "hanlp" ]
    then
        dataDir=data/cais_bies-${bies}_token-hanlp
    elif [ ${tokenizer} == "stanford" ]
    then
        dataDir=data/cais_bies-${bies}_token-stanford
    elif [ ${tokenizer} == "pkuseg" ]
    then
        dataDir=data/cais_bies-${bies}_token-pkuseg
    else
        echo "Incorrect tokenizer."
    fi

    # training command for cais dataset
    echo "Training our model with '${model_type}' mode on CAIS dataset."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py -dd ${dataDir} \
        -sd model/cais_bies-${bies}_token-${tokenizer}_u_wse_bs-16_dr-0.3_wed-128_ehd-512_aod-128_sed-32_sdhd-64_${partial_model_name} \
        -u -bs 16 -dr 0.3 -ced 128 -wed 128 -ehd 512 -aod 128 -sed 32 -sdhd 64 ${partial_command}
elif [ ${dataset_name} == "ecdt" ]
then
    bies=$4
    tokenizer=$5

    if [ ${tokenizer} == "ltp" ]
    then
        dataDir=data/ecdt_di-False_ds-False_for_stack_bies-${bies}_token-True
    elif [ ${tokenizer} == "jieba" ]
    then
        dataDir=data/ecdt_di-False_ds-False_for_stack_bies-${bies}_token-jieba
    elif [ ${tokenizer} == "hanlp" ]
    then
        dataDir=data/ecdt_di-False_ds-False_for_stack_bies-${bies}_token-hanlp
    elif [ ${tokenizer} == "stanford" ]
    then
        dataDir=data/ecdt_di-False_ds-False_for_stack_bies-${bies}_token-stanford
    elif [ ${tokenizer} == "pkuseg" ]
    then
        dataDir=data/ecdt_di-False_ds-False_for_stack_bies-${bies}_token-pkuseg
    else
        echo "Incorrect tokenizer."
    fi

    # training command for cais dataset
    echo "Training our model with '${model_type}' mode on ECDT dataset."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py -dd ${dataDir} \
        -sd model/ecdt_bies-${bies}_token-${tokenizer}_u_wse_bs-8_dr-0.3_wed-128_ehd-512_aod-128_sed-64_sdhd-64_${partial_model_name} \
        -valid_file test.txt \
        -u -bs 8 -dr 0.3 -ced 128 -wed 128 -ehd 512 -aod 128 -sed 64 -sdhd 64 ${partial_command}
else
    echo "Incorrect dataset name."
fi