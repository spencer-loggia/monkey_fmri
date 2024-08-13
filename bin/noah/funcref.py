#!/usr/bin/env python3
#
# TODO: Add proper documentation.
#
# This script is based on the anatomical preprocessing workflows from
# sMRIPrep.

from nipype.interfaces import image
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype.interfaces.base import traits
from nipype.interfaces.mixins import CopyHeaderInterface
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants.segmentation import (
    DenoiseImageInputSpec,
    DenoiseImage as _DI,
)
from nipype.interfaces.freesurfer import MRIConvert
from niworkflows.interfaces.freesurfer import PatchedLTAConvert as LTAConvert
from niworkflows.interfaces.freesurfer import StructuralReference
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.images import Conform, TemplateDimensions
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.utils.misc import add_suffix


# Copy the header for the input image to `DenoiseImage`.
class _DenoiseImageInputSpec(DenoiseImageInputSpec):
    copy_header = traits.Bool(True, usedefault=True)


class DenoiseImage(_DI, CopyHeaderInterface):
    input_spec = _DenoiseImageInputSpec
    _copy_header_map = {"output_image": "input_image"}


def init_funcref_wf(base_dir, output_dir, name="funcref_wf"):
    workflow = Workflow(name=name)
    workflow.base_dir = base_dir

    inputnode = pe.Node(niu.IdentityInterface(fields=["func_files"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["func_ref", "func_valid_list", "func_realign_xfm", "out_report"]
        ),
        name="outputnode",
    )

    func_ref_dimensions = pe.Node(TemplateDimensions(), name="func_ref_dimensions")

    func_sphinx_correct = pe.MapNode(
        MRIConvert(sphinx=True), iterfield="in_file", name="func_sphinx_correct"
    )
    func_flip = pe.MapNode(
        MRIConvert(in_orientation="RAS", out_orientation="RAI"),
        iterfield="in_file",
        name="func_flip",
    )

    # Denoise and reorient functional images to RAS and resample to a
    # common voxel space.
    denoise = pe.MapNode(
        DenoiseImage(noise_model="Rician"), iterfield="input_image", name="denoise"
    )
    func_conform = pe.MapNode(Conform(), iterfield="in_file", name="func_conform")

    workflow.connect(
        [
            (inputnode, func_ref_dimensions, [("func_files", "t1w_list")]),
            (func_ref_dimensions, func_sphinx_correct, [("t1w_valid_list", "in_file")]),
            (
                func_ref_dimensions,
                func_conform,
                [("target_zooms", "target_zooms"), ("target_shape", "target_shape")],
            ),
            (func_sphinx_correct, func_flip, [("out_file", "in_file")]),
            (func_flip, denoise, [("out_file", "input_image")]),
            (denoise, func_conform, [("output_image", "in_file")]),
            (
                func_ref_dimensions,
                outputnode,
                [("out_report", "out_report"), ("t1w_valid_list", "func_valid_list")],
            ),
        ]
    )

    func_conform_xfm = pe.MapNode(
        LTAConvert(in_lta="identity.nofile", out_lta=True),
        iterfield=["source_file", "target_file"],
        name="func_conform_xfm",
    )

    # Correct for intensity bias.
    n4_correct = pe.MapNode(
        N4BiasFieldCorrection(dimension=3, copy_header=True),
        iterfield="input_image",
        name="n4_correct",
        n_procs=1,
    )

    # Align and merge the functional images.
    func_merge = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            intensity_scaling=True,
            subsamble_threshold=200,
            transform_outputs=True,
        ),
        name="func_merge",
    )

    # Reorient the template to RAS if necessary.
    func_reorient = pe.Node(image.Reorient(), name="func_reorient")

    merge_xfm = pe.MapNode(
        niu.Merge(2),
        iterfield=["in1", "in2"],
        name="merge_xfm",
        run_without_submitting=True,
    )
    concat_xfms = pe.MapNode(
        ConcatenateXFMs(inverse=True),
        iterfield="in_xfms",
        name="concat_xfms",
        run_without_submitting=True,
    )

    workflow.connect(
        [
            (
                func_ref_dimensions,
                func_conform_xfm,
                [("t1w_valid_list", "source_file")],
            ),
            (func_conform, func_conform_xfm, [("out_file", "target_file")]),
            (func_conform, n4_correct, [("out_file", "input_image")]),
            (
                func_conform,
                func_merge,
                [(("out_file", add_suffix, "_template"), "out_file")],
            ),
            (n4_correct, func_merge, [("outpute_image", "in_files")]),
            (func_merge, func_reorient, [("out_file", "in_file")]),
            (func_conform_xfm, merge_xfm, [("out_lta", "in1")]),
            (func_merge, merge_xfm, [("transform_outputs", "in2")]),
            (merge_xfm, concat_xfms, [("out", "in_xfms")]),
            (func_reorient, outputnode, [("out_file", "func_ref")]),
            (concat_xfms, outputnode, [("out_xfm", "func_realign_xfm")]),
        ]
    )

    return workflow


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
