import unittest

from pi3.models.pi3_training import Pi3


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compact_model(**overrides):
    kwargs = dict(
        encoder_size="small",
        encoder_pretrained=False,
        decoder_size="small",
        head_dim=384,
        point_decoder_dim=384,
        point_decoder_heads=6,
        camera_decoder_dim=384,
        camera_decoder_heads=6,
        camera_head_dim=256,
        global_point_decoder_dim=384,
        global_point_decoder_heads=6,
        load_vggt=False,
        freeze_encoder=True,
        ckpt=None,
        num_dec_blk_not_to_checkpoint=999,
    )
    kwargs.update(overrides)
    return Pi3(**kwargs)


class Pi3CompactModelTest(unittest.TestCase):
    def test_compact_visibility_mask_model_instantiates_from_scratch(self):
        model = compact_model(use_visibility_mask_conditioning=True)

        self.assertEqual(model.dec_embed_dim, 384)
        self.assertEqual(model.visibility_mask_embed.proj.out_channels, 384)
        self.assertEqual(model.point_head.proj.in_features, 384)
        self.assertEqual(model.camera_head.fc_t.in_features, 256)
        self.assertFalse(any(param.requires_grad for param in model.encoder.parameters()))

        total = count_parameters(model)
        trainable = count_trainable_parameters(model)
        self.assertGreater(total, 80_000_000)
        self.assertLess(total, 120_000_000)
        self.assertLess(trainable, total)

    def test_compact_ray_and_mask_conditioning_share_decoder_dim(self):
        model = compact_model(
            use_ray_conditioning=True,
            use_visibility_mask_conditioning=True,
            ckpt="none",
        )

        self.assertEqual(model.ray_embed.proj.out_channels, model.dec_embed_dim)
        self.assertEqual(model.visibility_mask_embed.proj.out_channels, model.dec_embed_dim)

    def test_encoder_decoder_dim_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "encoder and decoder dimensions"):
            compact_model(decoder_size="base")

    def test_vggt_loading_is_restricted_to_large_pi3_shape(self):
        with self.assertRaisesRegex(ValueError, "load_vggt=true"):
            compact_model(load_vggt=True)

    def test_legacy_head_dim_alias_must_match_point_decoder_dim(self):
        with self.assertRaisesRegex(ValueError, "head_dim is a legacy alias"):
            compact_model(head_dim=512)


if __name__ == "__main__":
    unittest.main()
