python ./src/esd_meta_sana.py \
    --pretrained_model_name_or_path \
        "Efficient-Large-Model/SANA_Sprint_0.6B_1024px_teacher_diffusers" \
    --removing_concepts \
        "nudity" \
    --validation_prompts \
        "japan body" \
    --num_images_per_prompt 10 \
    --train_batch_size 10 \
    --guidance_scale 4.5 \
    --concept_scale 4.5 \
    --devices 0 0 \
    --num_train_steps 1500 \
    --finetuning_method full \
    --gamma1_1 0.1 \
    --gamma1_2 0.01 \
    --gamma2_1 0.1 \
    --gamma2_2 0.1 \
    --gamma2_3 0.01 \
    --seed 42 \
    --resolution 1024 \
    --use_wandb \
    --wandb_project meta_unlearning \
    --exp_name sana_esd_meta_unlearn_001

    # --fix_timesteps True \
    # --fixed_time_steps 1 2 5 10
