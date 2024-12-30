from postprocessor import Postprocessor as BasePostprocessor
from .utils.cv_utils.geo_utils import calc_iou


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.iou = None
        self.pre_n = 3
        self.pre_targets = []

    def _process(self, result, filter_result):
        hit = False
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        polygons = self._gen_polygons()
        model_name, rectangles = next(iter(filter_result.items()))
        for rectangle in rectangles:
            diff_num = 0
            for pre_targets_ in self.pre_targets:
                max_iou = 0
                for pre_target in pre_targets_:
                    iou = calc_iou(pre_target['xyxy'], rectangle['xyxy'])
                    max_iou = max(max_iou, iou)
                if max_iou > 0 and max_iou < self.iou:
                    diff_num += 1
                    break
            if diff_num and len(self.pre_targets) == self.pre_n:
                hit = True
                rectangle['color'] = self.alert_color
        if len(rectangles) > 0:
            self.pre_targets.append(rectangles)
        if len(self.pre_targets) > self.pre_n:
            self.pre_targets.pop(0)
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True
