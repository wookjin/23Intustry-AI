from durable.lang import *

with ruleset('Solution'):
    @when_all(+m.subject)

    def output(c):
        print('Fact: {0} {1} {2}'.format(c.m.subject, c.m.object, c.m.predicate))

    # 기본 정의
    @when_all((m.subject == '기체') & (m.object == '문제발생') & (m.predicate == '프로펠러 이상'))
    def categoryBodyCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': '프로펠러', 'predicate': '파손'})

    @when_all((m.subject == '기체') & (m.object == '문제발생') & (m.predicate == '모터 이상'))
    def categoryBodyCheckB(c):
        c.assert_fact({'subject': c.m.subject, 'object': '모터', 'predicate': '동작하지 않음'})

    @when_all((m.subject == '배터리') & (m.object == '문제발생') & (m.predicate == '배터리 이상'))
    def categoryBatteryCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': '배터리', 'predicate': '출력 50V 이하'})

    @when_all((m.subject == '날씨') & (m.object == '문제발생') & (m.predicate == '날씨 이상'))
    def categoryWeatherCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': '비가', 'predicate': '내리고 있다'})
        c.assert_fact({'subject': c.m.subject, 'object': '눈이', 'predicate': '내리고 있다'})
        c.assert_fact({'subject': c.m.subject, 'object': '평균풍속이', 'predicate': '초속 5m 이상 이다'})
        c.assert_fact({'subject': c.m.subject, 'object': '현재 기온이', 'predicate': '영하 10도 이하 이다'})

    @when_all((m.subject == 'GPS모듈') & (m.object == '문제발생') & (m.predicate == 'GPS 이상'))
    def categoryGPSCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': '연결된 GPS위성 개수가', 'predicate': '0개 이다'})

    @when_all((m.subject == '드론자세') & (m.object == '문제발생') & (m.predicate == '드론자세 이상'))
    def categoryPostureCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'Pitch각도가 ', 'predicate': '5도 이상이다'})
        c.assert_fact({'subject': c.m.subject, 'object': 'Roll각도가 ', 'predicate': '5도 이상이다'})

    @when_all((m.subject == 'LTE 라우터') & (m.object == '문제발생') & (m.predicate == 'LTE 이상'))
    def categoryPostureCheckA(c):
        c.assert_fact({'subject': c.m.subject, 'object': '라우터 전원', 'predicate': '켜짐'})


    # 프로펠러 이슈
    @when_all(c.first << (m.subject == '기체') & (m.object == '문제발생') & (m.predicate == '프로펠러 이상'), (m.object == '프로펠러') & (m.predicate == '파손'))
    def categoryPropellerA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '프로펠러', 'predicate': '교체한다'})

    # 모터 이슈
    @when_all(c.first << (m.subject == '기체') & (m.object == '문제발생') & (m.predicate == '모터 이상'),
              (m.object == '모터') & (m.predicate == '동작하지 않음'))
    def categoryMotorA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '소음', 'predicate': '발생함'})

    @when_all((m.object == '소음') & (m.predicate == '발생함'))
    def categoryMotorB(c):
        c.assert_fact({'subject': c.m.subject, 'object': '모터', 'predicate': '교체한다'})

    # 배터리 이슈
    @when_all(c.first << (m.subject == '배터리') & (m.object == '문제발생') & (m.predicate == '배터리 이상'),
              (m.object == '배터리') & (m.predicate == '출력 50V 이하'))
    def categoryBatteryA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '셀당 배터리가', 'predicate': '1V 이하 차이난다'})

    @when_all((m.object == '셀당 배터리가') & (m.predicate == '1V 이하 차이난다'))
    def categoryBatteryB(c):
        c.assert_fact({'subject': c.m.subject, 'object': '셀 밸런싱을', 'predicate': '확인한다'})

    @when_all((m.object == '셀 밸런싱을') & (m.predicate == '확인한다'))
    def categoryBatteryD(c):
        c.assert_fact({'subject': c.m.subject, 'object': '배터리를', 'predicate': '충전한다'})

    @when_all((m.object == '배터리를') & (m.predicate == '충전한다'))
    def categoryBatteryD(c):
        c.assert_fact({'subject': c.m.subject, 'object': '셀당 배터리가', 'predicate': '1V 이상 차이난다'})

    @when_all((m.object == '셀당 배터리가') & (m.predicate == '1V 이상 차이난다'))
    def categoryBatteryE(c):
        c.assert_fact({'subject': c.m.subject, 'object': '배터리', 'predicate': '교체한다'})

    # 날씨 이슈
    @when_any(c.first << (m.subject == '날씨') & (m.object == '문제발생') & (m.predicate == '날씨 이상'),
              (m.object == '눈이') & (m.predicate == '내리고 있다.'), (m.object == '비가') & (m.predicate == '내리고 있다.'),
              (m.object =='평균풍속이') & (m.predicate == '초속 5m 이상 이다'), (m.object =='현재 기온이') & (m.predicate == '영하 10도 이하 이다'))
    def categoryWeatherA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '이륙', 'predicate': '불가능'})

    # GPS 이슈
    @when_all(c.first << (m.subject == 'GPS모듈') & (m.object == '문제발생') & (m.predicate == 'GPS 이상'),
              (m.object == '연결된 GPS위성 개수가') & (m.predicate == '0개 이다'))
    def categoryGPSA(c):
        c.assert_fact({'subject': c.first.subject, 'object': 'GPS 모듈 전원을', 'predicate': '확인한다'})

    @when_all((m.object == 'GPS 모듈 전원을') & (m.predicate == '확인한다'))
    def categoryGPSB(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'GPS 모듈을', 'predicate': '교체한다'})

    @when_all((m.object == 'GPS 모듈을') & (m.predicate == '교체한다'))
    def categoryGPSC(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'FC를', 'predicate': '교체한다'})

    # 자세 상태 이슈
    @when_any(c.first << (m.subject == '드론자세') & (m.object == '문제발생') & (m.predicate == '드론자세 이상'),
              (m.object == 'Pitch각도가') & (m.predicate == '5도 이상인가'), (m.object == 'Roll각도가') & (m.predicate == '5도 이상인가'))
    def categoryPostureA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '드론기체를', 'predicate': '평지로 이동시킨다'})

    @when_all((m.object == '드론기체를') & (m.predicate == '평지로 이동시킨다'))
    def categoryPostureB(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'FC를', 'predicate': '교체한다'})

    @when_all((m.object == 'FC를') & (m.predicate == '교체한다'))
    def categoryPostureC(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'Yaw 각도가', 'predicate': '나침반 기준 10도 이상 차이가 난다'})

    @when_all((m.object == 'Yaw 각도가') & (m.predicate == '나침반 기준 10도 이상 차이가 난다'))
    def categoryPostureD(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'Compass Callibration을', 'predicate': '진행한다'})

    @when_all((m.object == 'Compass Callibration을') & (m.predicate == '진행한다'))
    def categoryPostureE(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'Compass 센서를', 'predicate': '교체한다'})

    # LTE 이슈
    @when_all(c.first << (m.subject == 'LTE 라우터') & (m.object == '문제발생') & (m.predicate == 'LTE 이상'),
              (m.object == '라우터 전원') & (m.predicate == '켜짐'))
    def categoryLTEA(c):
        c.assert_fact({'subject': c.first.subject, 'object': '접속', 'predicate': '안됨'})

    @when_all((m.object == '접속') & (m.predicate == '안됨'))
    def categoryLTEB(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'USIM을', 'predicate': '확인한다'})

    @when_all(c.first << (m.subject == 'LTE 라우터') & (m.object == '문제발생') & (m.predicate == 'LTE 이상'),
              (m.object == '라우터 전원') & (m.predicate == '켜짐'))
    def categoryLTEC(c):
        c.assert_fact({'subject': c.first.subject, 'object': '드론-pc간 데이터 송수신을', 'predicate': '확인한다'})

    @when_all((m.object == '드론-pc간 데이터 송수신을') & (m.predicate == '확인한다'))
    def categoryLTED(c):
        c.assert_fact({'subject': c.m.subject, 'object': 'LED상태 및 연결상태를', 'predicate': '확인한다'})

def exitByOutOfRange():
    print('입력 가능한 범위 초과')
    exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    checkItemType = int(input("Issue Item [Body(1)],[Battery(2)],[Weather(3)],[GPS(4)],[Posture(5)],[LTE(6)] :"))
    checkItemName = str
    if checkItemType == 1:
        checkItemName = '기체'
        BodyIssueType = int(input("기체확인 [프로펠러 이상(1)],[모터 이상(2)],[프로펠러와 모터 이상 없음(3)] :"))
        BodyIssueTitle = '문제발생'
        BodyIssueIsOK = str
        if BodyIssueType == 1:
            BodyIssueIsOK = '프로펠러 이상'
        elif BodyIssueType == 2:
            BodyIssueIsOK = '모터 이상'
        elif BodyIssueType == 3:
            BodyIssueTitle = '프로펠러와 모터는'
            BodyIssueIsOK = '문제 없다'
        else:
            exitByOutOfRange()

        assert_fact('Solution', {'subject': checkItemName, 'object': BodyIssueTitle, 'predicate': BodyIssueIsOK})

    elif checkItemType == 2:
        checkItemName = '배터리'
        BatteryIssueType = int(input("배터리 출력확인 [출력 50V 이하(1)],[출력 50V 이상(2)] : "))
        BatteryIssueTitle = '문제발생'
        BatteryIssueIsOK = str
        if BatteryIssueType == 1:
            BatteryIssueIsOK = '배터리 이상'
        elif BatteryIssueType == 2:
            BatteryIssueTitle = '비행'
            BatteryIssueIsOK = '가능'
        else :
            exitByOutOfRange()

        assert_fact('Solution', {'subject': checkItemName, 'object': BatteryIssueTitle, 'predicate': BatteryIssueIsOK})

    elif checkItemType == 3:
        checkItemName = '날씨'
        assert_fact('Solution', {'subject': checkItemName, 'object': '문제발생', 'predicate': '날씨 이상'})
        
    elif checkItemType == 4:
        checkItemName = 'GPS모듈'
        assert_fact('Solution', {'subject': checkItemName, 'object': '문제발생', 'predicate': 'GPS 이상'})

    elif checkItemType == 5:
        checkItemName = '드론자세'
        assert_fact('Solution', {'subject': checkItemName, 'object': '문제발생', 'predicate': '드론자세 이상'})

    elif checkItemType == 6:
        checkItemName = 'LTE 라우터'
        LTEIssueType = int(input("LTE 전원확인 [전원 켜짐(1)],[전원 안켜짐(2)] : "))
        LTEIssueTitle = '문제발생'
        LTEIssueIsOK = str
        if LTEIssueType == 1:
            LTEIssueIsOK = 'LTE 이상'
        elif LTEIssueType == 2:
            LTEIssueTitle = ''
            LTEIssueIsOK = '교체한다'
        else:
            exitByOutOfRange()

        assert_fact('Solution', {'subject': checkItemName, 'object': LTEIssueTitle, 'predicate': LTEIssueIsOK})
    else:
        exitByOutOfRange()

    #assert_fact('Solution', {'subject': checkItemName, 'object': '문제가', 'predicate': '발생 했다'})
