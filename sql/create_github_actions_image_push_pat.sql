-- create_github_actions_image_push_pat.sql
-- Bootstrap a Snowflake service user and Programmatic Access Token (PAT)
-- for GitHub Actions to push the AutoGluon SPCS image to the Snowflake image registry.
--
-- IMPORTANT:
--   1. Run this in Snowsight/SnowSQL with a role that can create users, roles,
--      grant image repository privileges, and modify programmatic authentication methods.
--   2. The PAT token_secret is shown only once in the ALTER USER ... ADD PAT result.
--      Copy it immediately into the GitHub repository secret:
--        SNOWFLAKE_REGISTRY_PASSWORD
--   3. The GitHub workflow uses this PAT as the Snowflake CLI password for
--      GITHUB_ACTIONS_IMAGE_PUSHER, then Snowflake CLI logs Docker in with a
--      short-lived image-registry token.
--
-- If SECURITYADMIN cannot create authentication policies in TABPFN_SCHEMA, run
-- this once with ACCOUNTADMIN or another schema owner:
--
--   USE ROLE ACCOUNTADMIN;
--   GRANT USAGE ON DATABASE TABPFN_DB TO ROLE SECURITYADMIN;
--   GRANT USAGE ON SCHEMA TABPFN_DB.TABPFN_SCHEMA TO ROLE SECURITYADMIN;
--   GRANT CREATE AUTHENTICATION POLICY ON SCHEMA TABPFN_DB.TABPFN_SCHEMA
--     TO ROLE SECURITYADMIN;

USE ROLE SECURITYADMIN;
USE DATABASE TABPFN_DB;
USE SCHEMA TABPFN_SCHEMA;

CREATE ROLE IF NOT EXISTS TABPFN_IMAGE_PUSHER;

CREATE USER IF NOT EXISTS GITHUB_ACTIONS_IMAGE_PUSHER
  TYPE = SERVICE
  LOGIN_NAME = 'GITHUB_ACTIONS_IMAGE_PUSHER'
  COMMENT = 'GitHub Actions service user for pushing SPCS AutoGluon images';

GRANT ROLE TABPFN_IMAGE_PUSHER TO USER GITHUB_ACTIONS_IMAGE_PUSHER;

ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  SET DEFAULT_ROLE = 'TABPFN_IMAGE_PUSHER';

GRANT USAGE ON DATABASE TABPFN_DB TO ROLE TABPFN_IMAGE_PUSHER;
GRANT USAGE ON SCHEMA TABPFN_DB.TABPFN_SCHEMA TO ROLE TABPFN_IMAGE_PUSHER;
GRANT READ, WRITE ON IMAGE REPOSITORY TABPFN_DB.TABPFN_SCHEMA.AUTOGLUON_IMAGE_REPOSITORY
  TO ROLE TABPFN_IMAGE_PUSHER;

-- PAT network policy requirement:
-- Snowflake service users require a network policy to create/use PATs by default.
-- GitHub-hosted runners do not have stable source IPs, so this script applies a
-- user-scoped authentication policy that removes the network-policy requirement
-- only for this service user. The PAT remains role-restricted to TABPFN_IMAGE_PUSHER.
--
-- Stricter alternative for self-hosted runners:
--   Create a network policy allowing only your runner's static outbound IP/CIDR,
--   then set it on GITHUB_ACTIONS_IMAGE_PUSHER instead of using this auth policy.
CREATE AUTHENTICATION POLICY IF NOT EXISTS TABPFN_GITHUB_ACTIONS_PAT_POLICY
  PAT_POLICY = (
    NETWORK_POLICY_EVALUATION = ENFORCED_NOT_REQUIRED
    REQUIRE_ROLE_RESTRICTION_FOR_SERVICE_USERS = TRUE
  )
  COMMENT = 'Allow role-restricted PATs for GitHub Actions image push service user';

ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  SET AUTHENTICATION POLICY TABPFN_GITHUB_ACTIONS_PAT_POLICY;

-- Create a short-lived PAT for Docker registry login from GitHub Actions.
-- Copy the token_secret from the result immediately; it will not be shown again.
ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
  ADD PROGRAMMATIC ACCESS TOKEN GITHUB_ACTIONS_DOCKER_PUSH
  ROLE_RESTRICTION = 'TABPFN_IMAGE_PUSHER'
  DAYS_TO_EXPIRY = 90
  COMMENT = 'GitHub Actions Docker login for Snowflake image registry';

-- GitHub repository secrets:
--   SNOWFLAKE_REGISTRY_PASSWORD = <token_secret from command above>

-- Verify PAT metadata later. This does not reveal token_secret.
SHOW USER PROGRAMMATIC ACCESS TOKENS FOR USER GITHUB_ACTIONS_IMAGE_PUSHER;

-- Verify authentication policy assignment:
-- DESC USER GITHUB_ACTIONS_IMAGE_PUSHER;

-- Verify image repository access after workflow push:
-- SHOW IMAGES IN IMAGE REPOSITORY TABPFN_DB.TABPFN_SCHEMA.AUTOGLUON_IMAGE_REPOSITORY;

-- Revoke/rotate procedure:
-- ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
--   REMOVE PROGRAMMATIC ACCESS TOKEN GITHUB_ACTIONS_DOCKER_PUSH;
--
-- ALTER USER GITHUB_ACTIONS_IMAGE_PUSHER
--   ADD PROGRAMMATIC ACCESS TOKEN GITHUB_ACTIONS_DOCKER_PUSH
--   ROLE_RESTRICTION = 'TABPFN_IMAGE_PUSHER'
--   DAYS_TO_EXPIRY = 90
--   COMMENT = 'GitHub Actions Docker login for Snowflake image registry';
