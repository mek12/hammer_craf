"""Standalone CPU checks for v2; no encoders, downloads, or HAMMER imports."""

import inspect
import itertools
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from craf_residual import FrozenCRAFResidual as V1
from craf_residual_v2 import FrozenCRAFResidual, correction_penalty


class FrozenCRAFResidualV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.text = torch.randn(4, 6)
        self.image = torch.randn(4, 6)
        self.baseline = torch.randn(4, 2)

    def make_head(self, **kwargs):
        return FrozenCRAFResidual(feature_dim=6, hidden_dim=4, **kwargs)

    def test_constructor_defaults_and_state_shapes(self):
        old_parameters = inspect.signature(V1).parameters
        new_parameters = inspect.signature(FrozenCRAFResidual).parameters
        self.assertEqual(list(new_parameters), [*old_parameters, "condition_on_decision"])
        for name, parameter in old_parameters.items():
            self.assertEqual(parameter.default, new_parameters[name].default)
        self.assertIs(new_parameters["condition_on_decision"].default, False)
        old = V1()
        new = FrozenCRAFResidual()
        self.assertEqual(
            {key: value.shape for key, value in old.state_dict().items()},
            {key: value.shape for key, value in new.state_dict().items()},
        )
        new.load_state_dict(old.state_dict(), strict=True)
        old.load_state_dict(new.state_dict(), strict=True)

    def test_exact_v1_forward_and_gradients_all_ablations(self):
        for flags in itertools.product((True, False), repeat=3):
            options = dict(zip(("use_reliability", "use_consistency", "use_interactions"), flags))
            old = V1(6, 4, dropout=.2, **options)
            # Nonzero correction ensures fidelity is not hidden by initialization.
            with torch.no_grad():
                old.classifier.net[3].weight.normal_()
                old.classifier.net[3].bias.normal_()
            new = self.make_head(dropout=.2, **options)
            new.load_state_dict(old.state_dict(), strict=True)
            for training in (False, True):
                with self.subTest(flags=flags, training=training):
                    old.train(training)
                    new.train(training)
                    torch.manual_seed(123)
                    expected, expected_details = old(self.text, self.image, self.baseline, True)
                    torch.manual_seed(123)
                    actual, details = new(self.text, self.image, self.baseline, True)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertIsNone(details["decision_features"])
                    for name, value in expected_details.items():
                        if value is None:
                            self.assertIsNone(details[name])
                        else:
                            self.assertTrue(torch.equal(details[name], value), name)
                    old.zero_grad(set_to_none=True)
                    new.zero_grad(set_to_none=True)
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    for (name, p_old), (_, p_new) in zip(old.named_parameters(), new.named_parameters()):
                        if p_old.grad is None:
                            self.assertIsNone(p_new.grad, name)
                        else:
                            self.assertTrue(torch.equal(p_old.grad, p_new.grad), name)

    def test_conditioning_changes_only_classifier_input_shape(self):
        off = self.make_head()
        on = self.make_head(condition_on_decision=True)
        self.assertEqual(off.state_dict().keys(), on.state_dict().keys())
        for name, value in off.state_dict().items():
            expected = (4, 24) if name == "classifier.net.0.weight" else value.shape
            self.assertEqual(on.state_dict()[name].shape, expected, name)
        self.assertEqual(off.classifier.net[0].in_features, 21)
        self.assertEqual(on.classifier.net[0].in_features, 24)

    def test_zero_parity_and_gate_semantics_when_conditioned(self):
        for flags in itertools.product((True, False), repeat=3):
            options = dict(zip(("use_reliability", "use_consistency", "use_interactions"), flags))
            off = self.make_head(dropout=0, **options)
            on = self.make_head(dropout=0, condition_on_decision=True, **options)
            # Copy compatible state so the gates and projected embeddings match.
            state = on.state_dict()
            state.update({k: v for k, v in off.state_dict().items() if k != "classifier.net.0.weight"})
            on.load_state_dict(state, strict=True)
            for training in (False, True):
                with self.subTest(flags=flags, training=training):
                    on.train(training)
                    _, old_details = off(self.text, self.image, self.baseline, True)
                    logits, details = on(self.text, self.image, self.baseline, True)
                    self.assertTrue(torch.equal(logits, self.baseline))
                    self.assertEqual(torch.count_nonzero(details["residual_logits"]).item(), 0)
                    self.assertTrue(torch.equal(details["fused_embedding"][:, :-3], old_details["fused_embedding"]))
                    self.assertTrue(torch.equal(details["fused_embedding"][:, -3:], details["decision_features"]))
                    for name, value in old_details.items():
                        if name in ("fused_embedding", "residual_logits", "decision_features"):
                            continue
                        if value is None:
                            self.assertIsNone(details[name])
                        else:
                            self.assertTrue(torch.equal(details[name], value), name)

    def test_decision_formula_and_detachment(self):
        head = self.make_head(dropout=0, condition_on_decision=True)
        baseline = self.baseline.clone().requires_grad_()
        with torch.no_grad():
            head.classifier.net[3].weight.normal_()
        output, details = head(self.text, self.image, baseline, True)
        probabilities = baseline.detach().softmax(dim=-1)
        margin = baseline.detach()[:, 1] - baseline.detach()[:, 0]
        expected = torch.stack((
            probabilities[:, 1], margin / (1 + margin.abs()),
            -(probabilities * baseline.detach().log_softmax(dim=-1)).sum(dim=-1) / math.log(2),
        ), dim=-1)
        features = details["decision_features"]
        self.assertTrue(torch.equal(features, expected))
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad_fn)
        self.assertIsNone(torch.autograd.grad(
            details["residual_logits"].sum(), baseline, allow_unused=True, retain_graph=True
        )[0])
        self.assertTrue(torch.equal(torch.autograd.grad(output.sum(), baseline)[0], torch.ones_like(baseline)))

    def test_extreme_finite_logits_and_entropy(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                head = self.make_head(dropout=0, condition_on_decision=True).to(dtype=dtype)
                limit = torch.finfo(dtype).max
                baseline = torch.tensor([[0, 0], [-limit, limit], [limit, -limit], [limit, limit]], dtype=dtype)
                try:
                    output, details = head(self.text.to(dtype), self.image.to(dtype), baseline, True)
                except RuntimeError as error:
                    if dtype in (torch.float16, torch.bfloat16) and 'not implemented' in str(error):
                        self.skipTest(f'This PyTorch CPU build does not support {dtype}: {error}')
                    raise
                features = details["decision_features"]
                self.assertTrue(torch.isfinite(features).all().item())
                self.assertTrue(torch.equal(output, baseline))
                expected = torch.tensor([[.5, 0, 1], [1, 1, 0], [0, -1, 0], [.5, 0, 1]], dtype=dtype)
                torch.testing.assert_close(features, expected)
                self.assertTrue((features[:, 0] >= 0).all().item())
                self.assertTrue((features[:, 0] <= 1).all().item())
                self.assertTrue((features[:, 1].abs() <= 1).all().item())

    def test_common_logit_shift_preserves_features_and_correction(self):
        head = self.make_head(dropout=0, condition_on_decision=True).double().eval()
        with torch.no_grad():
            head.classifier.net[3].weight.normal_()
        baseline = torch.tensor([[1, 3], [-4, 0], [2, 2], [7, -1]], dtype=torch.float64)
        shifts = torch.tensor([[100], [-20], [8], [32]], dtype=torch.float64)
        args = self.text.double(), self.image.double()
        _, first = head(*args, baseline, True)
        _, shifted = head(*args, baseline + shifts, True)
        self.assertTrue(torch.equal(first["decision_features"], shifted["decision_features"]))
        self.assertTrue(torch.equal(first["residual_logits"], shifted["residual_logits"]))

    def test_penalty_common_mode_and_gradient(self):
        common = torch.tensor([[3., 3.], [-8., -8.]], requires_grad=True)
        loss = correction_penalty(common)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(common.grad, torch.zeros_like(common)))
        residual = torch.tensor([[1., 3.], [-4., 2.]], requires_grad=True)
        penalty = correction_penalty(residual)
        self.assertEqual(penalty.item(), 20)
        self.assertEqual(correction_penalty(residual + torch.tensor([[10.], [-20.]])).item(), 20)
        penalty.backward()
        self.assertTrue(torch.equal(residual.grad, torch.tensor([[-2., 2.], [-6., 6.]])))
        self.assertTrue(torch.equal(residual.grad.sum(dim=-1), torch.zeros(2)))

    def test_learning_updates_conditioned_path(self):
        head = self.make_head(dropout=0, condition_on_decision=True)
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        target = torch.tensor([0, 1, 0, 1])
        baseline = self.baseline.clone().requires_grad_()
        text = self.text.clone().requires_grad_()
        image = self.image.clone().requires_grad_()
        initial_conditioned = head.classifier.net[0].weight[:, -3:].detach().clone()
        initial_final = head.classifier.net[3].weight.detach().clone()
        initial_loss = nn.functional.cross_entropy(head(text, image, baseline), target)
        initial_loss.backward()
        self.assertEqual(head.classifier.net[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial_final, head.classifier.net[3].weight))
        optimizer.zero_grad(set_to_none=True)
        text.grad = image.grad = baseline.grad = None
        logits, details = head(text, image, baseline, True)
        loss = nn.functional.cross_entropy(logits, target) + .1 * correction_penalty(details["residual_logits"])
        self.assertLess(loss.item(), initial_loss.item())
        loss.backward()
        self.assertGreater(head.classifier.net[0].weight.grad[:, -3:].abs().sum().item(), 0)
        for features in (text, image, baseline):
            self.assertGreater(features.grad.abs().sum().item(), 0)
        for module in (head.text_projection, head.image_projection, head.text_reliability,
                       head.image_reliability, head.consistency_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial_conditioned, head.classifier.net[0].weight[:, -3:]))


if __name__ == "__main__":
    unittest.main()
