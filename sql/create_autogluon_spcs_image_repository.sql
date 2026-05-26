-- create_autogluon_spcs_image_repository.sql
-- Run once to set up the Snowflake Image Repository for the custom AutoGluon SPCS image.

CREATE IMAGE REPOSITORY IF NOT EXISTS AUTOGLUON_IMAGE_REPOSITORY;

-- Verify the repository was created and get the repository URL:
SHOW IMAGE REPOSITORIES;

-- After building and tagging the Docker image locally, push it:
--
--   docker build --platform linux/amd64 \
--     -f docker/autogluon/Dockerfile \
--     -t tabpfn-autogluon-ray:1.0.0 .
--
--   docker login <account>.registry.snowflakecomputing.com
--
--   docker tag tabpfn-autogluon-ray:1.0.0 \
--     <repository_url>/tabpfn-autogluon-ray:1.0.0
--
--   docker push <repository_url>/tabpfn-autogluon-ray:1.0.0

-- Verify the image is available:
SHOW IMAGES IN IMAGE REPOSITORY AUTOGLUON_IMAGE_REPOSITORY;
