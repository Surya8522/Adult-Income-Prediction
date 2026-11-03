import mlflow
from mlflow.tracking import MlflowClient


# Set SQLite tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def automate_champion_challenger():

    print("[INFO] Starting Automated Model Lifecycle Manager...")

    client = MlflowClient()

    # Adult Income registered model
    model_name = "Adult_Income_Production_Model"

    # F1-score is the finalized metric
    # Higher F1-score is better
    metric_to_optimize = "f1_score"

    try:

        # =====================================================
        # 1. Discover New Models and Move Them to Staging
        # =====================================================

        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        new_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "None"
        ]

        if new_models:

            print(
                f"\n[INFO] Found {len(new_models)} "
                "new model(s). Moving them to Staging..."
            )

            for mv in new_models:

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Staging",
                    archive_existing_versions=False
                )

                print(
                    f"  -> Version {mv.version} "
                    "is now in Staging."
                )

        # =====================================================
        # Refresh Registry Data
        # =====================================================

        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        staging_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "Staging"
        ]

        if not staging_models:

            print(
                "\n[INFO] No models in Staging "
                "to evaluate. Exiting."
            )

            return

        # =====================================================
        # 2. Evaluate Staging Models
        # =====================================================

        print(
            "\n[INFO] Evaluating Staging candidates..."
        )

        best_challenger = None

        # Higher F1-score is better
        best_challenger_score = -float("inf")

        for mv in staging_models:

            run = client.get_run(
                mv.run_id
            )

            score = run.data.metrics.get(
                metric_to_optimize,
                -float("inf")
            )

            print(
                f"  -> Staging Candidate: "
                f"Version {mv.version} | "
                f"{metric_to_optimize}: "
                f"{score:.4f}"
            )

            if score > best_challenger_score:

                best_challenger_score = score

                best_challenger = mv

        print(
            f"\n[INFO] Best Challenger is "
            f"Version {best_challenger.version} "
            f"({metric_to_optimize}: "
            f"{best_challenger_score:.4f})"
        )

        # =====================================================
        # 3. Find Current Champion in Production
        # =====================================================

        production_models = [
            mv
            for mv in all_versions
            if mv.current_stage == "Production"
        ]

        current_champion = (
            production_models[0]
            if production_models
            else None
        )

        promote_challenger = False

        # =====================================================
        # 4. Champion vs Challenger
        # =====================================================

        if not current_champion:

            print(
                "[INFO] No model currently in Production. "
                "Challenger wins by default!"
            )

            promote_challenger = True

        else:

            champ_run = client.get_run(
                current_champion.run_id
            )

            champ_score = champ_run.data.metrics.get(
                metric_to_optimize,
                -float("inf")
            )

            print(
                f"[INFO] Current Production Champion: "
                f"Version {current_champion.version} | "
                f"{metric_to_optimize}: "
                f"{champ_score:.4f}"
            )

            # =================================================
            # Higher F1-score wins
            # =================================================

            if best_challenger_score > champ_score:

                print(
                    f"[SUCCESS] Challenger beat "
                    f"the Champion! "
                    f"({best_challenger_score:.4f} > "
                    f"{champ_score:.4f})"
                )

                promote_challenger = True

            else:

                print(
                    f"[INFO] Challenger failed to beat "
                    f"Champion "
                    f"({best_challenger_score:.4f} <= "
                    f"{champ_score:.4f}). "
                    f"Champion retains title."
                )

        # =====================================================
        # 5. Execute Lifecycle Promotion
        # =====================================================

        if promote_challenger:

            print(
                f"\n[INFO] Promoting Version "
                f"{best_challenger.version} "
                "to Production..."
            )

            client.transition_model_version_stage(
                name=model_name,
                version=best_challenger.version,
                stage="Production",
                archive_existing_versions=False
            )

            print(
                f"[SUCCESS] Version "
                f"{best_challenger.version} "
                "is now in Production."
            )

            # Archive previous Champion
            if current_champion:

                print(
                    f"[INFO] Archiving defeated Champion "
                    f"(Version {current_champion.version})..."
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

                print(
                    f"[SUCCESS] Version "
                    f"{current_champion.version} "
                    "has been archived."
                )

        # =====================================================
        # 6. Clean Up Remaining Staging Models
        # =====================================================

        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        remaining_staging = [
            mv
            for mv in final_versions
            if mv.current_stage == "Staging"
        ]

        if remaining_staging:

            print(
                "\n[INFO] Cleaning up Staging area..."
            )

            for mv in remaining_staging:

                print(
                    f"  -> Archiving Version "
                    f"{mv.version} "
                    "(Lost the competition)"
                )

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False
                )

        # =====================================================
        # 7. Final Status
        # =====================================================

        print(
            "\n[SUCCESS] Automated Model Lifecycle "
            "execution complete!"
        )

    except Exception as e:

        print(
            f"[ERROR] Failed to automate lifecycle: {e}"
        )


if __name__ == "__main__":
    automate_champion_challenger()