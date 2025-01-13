import os

from kiutils.schematic import Schematic

from app.services.layout_service.sch_analysis.analysis_sch import SchModel


def generate_mudules(sch_file_path='../data/temp/project/Project.kicad_sch'):
    """根据原理图生成模块"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sch_file_path = os.path.join(base_dir, sch_file_path)
    print(sch_file_path)

    sch_model = SchModel(sch_file_path)
    sch_model.analysis_graph_base_models_main()

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    print("test")


if __name__ == '__main__':
    generate_mudules()