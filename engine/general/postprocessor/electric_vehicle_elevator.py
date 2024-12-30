from postprocessor import Postprocessor as BasePostprocessor


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        model_name, rectangles = next(iter(filter_result.items()))
        for rectangle in rectangles:
            if rectangle['label'] in self.alert_label:
                hit = True
                rectangle['color'] = self.alert_color
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True
