-- Durable proposal confirmation and single-use leave requests.
-- Existing leave_requests rows remain valid with proposal_id = NULL.

ALTER TABLE action_proposals
    ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS confirmed_request_id TEXT,
    ADD COLUMN IF NOT EXISTS confirmed_idempotency_key TEXT,
    ADD COLUMN IF NOT EXISTS confirmed_audit_reference TEXT;

ALTER TABLE leave_requests
    ADD COLUMN IF NOT EXISTS proposal_id TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'leave_requests_proposal_fk'
          AND conrelid = 'leave_requests'::regclass
    ) THEN
        ALTER TABLE leave_requests
            ADD CONSTRAINT leave_requests_proposal_fk
            FOREIGN KEY (proposal_id)
            REFERENCES action_proposals(proposal_id);
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS leave_requests_proposal_uidx
    ON leave_requests (proposal_id)
    WHERE proposal_id IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'action_proposals_confirmation_state_ck'
          AND conrelid = 'action_proposals'::regclass
    ) THEN
        ALTER TABLE action_proposals
            ADD CONSTRAINT action_proposals_confirmation_state_ck
            CHECK (
                (
                    confirmed_at IS NULL
                    AND confirmed_request_id IS NULL
                    AND confirmed_idempotency_key IS NULL
                    AND confirmed_audit_reference IS NULL
                )
                OR
                (
                    confirmed_at IS NOT NULL
                    AND confirmed_request_id IS NOT NULL
                    AND confirmed_idempotency_key IS NOT NULL
                    AND confirmed_audit_reference IS NOT NULL
                )
            );
    END IF;
END
$$;
