from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from src.utils.thunder import resolve_thunder_instance_id, snapshot_and_delete_thunder_instance


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> object:
        return self._payload


class _FakeRequestsModule:
    class RequestException(Exception):
        pass

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("No fake responses left for requests.get")
        return self._responses.pop(0)

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("No fake responses left for requests.post")
        return self._responses.pop(0)


class ThunderUtilsTests(unittest.TestCase):
    def test_resolve_thunder_instance_id_prefers_env_over_hostname(self) -> None:
        with patch("src.utils.thunder.subprocess.check_output", return_value="Ashwins-MacBook-Pro.local\n"):
            instance_id, source = resolve_thunder_instance_id(
                {"TNR_INSTANCE_ID": "vq9089t3"}
            )

        self.assertEqual(instance_id, "vq9089t3")
        self.assertEqual(source, "TNR_INSTANCE_ID")

    def test_resolve_thunder_instance_id_ignores_non_instance_local_hostname(self) -> None:
        with patch("src.utils.thunder.subprocess.check_output", return_value="Ashwins-MacBook-Pro.local\n"):
            with patch("builtins.input", return_value="vq9089t3"):
                instance_id, source = resolve_thunder_instance_id({})

        self.assertEqual(instance_id, "vq9089t3")
        self.assertEqual(source, "console")

    def test_resolve_thunder_instance_id_ignores_plain_word_hostname_token(self) -> None:
        with patch(
            "src.utils.thunder.subprocess.check_output",
            return_value="thunder-instance-yla07bfi\n",
        ):
            instance_id, source = resolve_thunder_instance_id({})

        self.assertEqual(instance_id, "yla07bfi")
        self.assertEqual(source, "hostname")

    def test_delete_404_instance_not_found_is_treated_as_idempotent_success(self) -> None:
        fake_requests = _FakeRequestsModule(
            responses=[
                _FakeResponse(
                    status_code=200,
                    payload={
                        "0": {
                            "uuid": "vq9089t3",
                            "name": "vq9089t3",
                            "status": "RUNNING",
                        }
                    },
                ),
                _FakeResponse(status_code=202, payload={"accepted": True}),
                _FakeResponse(
                    status_code=404,
                    payload={
                        "error": "instance_not_found",
                        "message": "Instance not found or not supported for deletion",
                        "code": 404,
                    },
                ),
            ]
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            response = snapshot_and_delete_thunder_instance(
                instance_id="vq9089t3",
                api_key="thunder-test-key",
                create_snapshot=True,
                snapshot_name="autosnap-vq9089t3-test",
            )

        self.assertEqual(len(fake_requests.calls), 3)
        self.assertEqual(fake_requests.calls[0]["method"], "GET")
        self.assertTrue(fake_requests.calls[0]["url"].endswith("/v1/instances/list"))
        self.assertTrue(fake_requests.calls[1]["url"].endswith("/v1/snapshots/create"))
        self.assertEqual(fake_requests.calls[1]["json"]["instanceId"], "0")
        self.assertTrue(
            fake_requests.calls[2]["url"].endswith("/v1/instances/0/delete")
        )
        self.assertEqual(response["snapshot_name"], "autosnap-vq9089t3-test")
        self.assertTrue(response["create_snapshot"])
        self.assertEqual(response["instance_ref"], "vq9089t3")
        self.assertEqual(response["resolved_instance_id"], "0")
        self.assertEqual(response["delete_status_code"], 404)
        self.assertTrue(response["delete_already_absent"])
        self.assertEqual(
            response["delete_response"]["error"],
            "instance_not_found",
        )

    def test_delete_other_404_still_raises(self) -> None:
        fake_requests = _FakeRequestsModule(
            responses=[
                _FakeResponse(
                    status_code=200,
                    payload={
                        "0": {
                            "uuid": "vq9089t3",
                            "name": "vq9089t3",
                            "status": "RUNNING",
                        }
                    },
                ),
                _FakeResponse(status_code=202, payload={"accepted": True}),
                _FakeResponse(
                    status_code=404,
                    payload={
                        "error": "different_error",
                        "message": "unexpected delete failure",
                        "code": 404,
                    },
                ),
            ]
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaises(RuntimeError) as ctx:
                snapshot_and_delete_thunder_instance(
                    instance_id="vq9089t3",
                    api_key="thunder-test-key",
                    create_snapshot=True,
                    snapshot_name="autosnap-vq9089t3-test",
                )

        self.assertIn("Thunder API delete instance failed", str(ctx.exception))

    def test_delete_can_run_without_snapshot_by_default(self) -> None:
        fake_requests = _FakeRequestsModule(
            responses=[
                _FakeResponse(
                    status_code=200,
                    payload={
                        "0": {
                            "uuid": "vq9089t3",
                            "name": "vq9089t3",
                            "status": "RUNNING",
                        }
                    },
                ),
                _FakeResponse(status_code=200, payload={"deleted": True}),
            ]
        )

        with patch.dict(sys.modules, {"requests": fake_requests}):
            response = snapshot_and_delete_thunder_instance(
                instance_id="vq9089t3",
                api_key="thunder-test-key",
            )

        self.assertEqual(len(fake_requests.calls), 2)
        self.assertEqual(fake_requests.calls[0]["method"], "GET")
        self.assertEqual(fake_requests.calls[1]["method"], "POST")
        self.assertTrue(fake_requests.calls[1]["url"].endswith("/v1/instances/0/delete"))
        self.assertFalse(response["create_snapshot"])
        self.assertIsNone(response["snapshot_name"])
        self.assertIsNone(response["snapshot_status_code"])


if __name__ == "__main__":
    unittest.main()
