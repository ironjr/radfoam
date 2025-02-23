import torch


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None


class TraceRays(torch.autograd.Function):
    _latest_ctx = None
    
    @staticmethod
    def forward(
        ctx,
        pipeline,
        _points,
        _attributes,
        _point_adjacency,
        _point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        weight_contribution=None,
        return_statistics=False,
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.depth_quantiles = depth_quantiles
        ctx.pipeline = pipeline
        ctx.points = _points
        ctx.attributes = _attributes
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets
        ctx.return_statistics = return_statistics

        if return_statistics:
            TraceRays._latest_ctx = ctx

        results = pipeline.trace_forward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles=depth_quantiles,
            weight_contribution=weight_contribution,
        )
        ctx.rgba = results["rgba"]
        ctx.depth_indices = results.get("depth_indices", None)

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["rgba"],
            results.get("depth", None),
            results.get("contribution", None),
            results["num_intersections"],
            errbox,
        )

    @staticmethod
    def backward(
        ctx,
        grad_rgba,
        grad_depth,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
    ):
        del grad_contribution
        del grad_num_intersections
        del errbox_grad

        rays = ctx.rays
        start_point = ctx.start_point
        pipeline = ctx.pipeline
        rgba = ctx.rgba
        _points = ctx.points
        _attributes = ctx.attributes
        _point_adjacency = ctx.point_adjacency
        _point_adjacency_offsets = ctx.point_adjacency_offsets
        depth_quantiles = ctx.depth_quantiles
        return_statistics = ctx.return_statistics

        results = pipeline.trace_backward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            rgba,
            grad_rgba,
            depth_quantiles,
            ctx.depth_indices,
            grad_depth,
            ctx.errbox.ray_error,
            return_grad_stats=return_statistics,
        )
        points_grad = results["points_grad"]
        attr_grad = results["attr_grad"]
        ctx.errbox.point_error = results.get("point_error", None)

        # Get gradient statistics if computed
        if ctx.return_statistics:
            ctx.point_grad_m2 = results["point_grad_m2"]
            ctx.attr_grad_m2 = results["attr_grad_m2"]
            ctx.point_grad_counts = results["point_grad_counts"]

        # Zero out non-finite gradients
        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.pipeline,
            ctx.rgba,
            ctx.points,
            ctx.attributes,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
            ctx.depth_quantiles,
        )
        return (
            None,  # pipeline
            points_grad,  # _points
            attr_grad,  # _attributes
            None,  # _point_adjacency
            None,  # _point_adjacency_offsets
            None,  # rays
            None,  # start_point
            None,  # depth_quantiles
            None,  # return_contribution
            None,  # return_statistics
        )

    @staticmethod
    def get_gradient_statistics(ctx=None):
        """Helper method to access gradient statistics after backward pass"""
        if ctx is None:
            ctx = TraceRays._latest_ctx
        if not hasattr(ctx, 'return_statistics') or not ctx.return_statistics:
            return None
        
        # Zero out non-finite gradients
        point_grad_m2 = ctx.point_grad_m2
        attr_grad_m2 = ctx.attr_grad_m2
        point_grad_m2[~point_grad_m2.isfinite()] = 0
        attr_grad_m2[~attr_grad_m2.isfinite()] = 0
        return {
            "point_grad_m2": point_grad_m2,
            "attr_grad_m2": attr_grad_m2,
            "point_grad_counts": ctx.point_grad_counts,
        }
